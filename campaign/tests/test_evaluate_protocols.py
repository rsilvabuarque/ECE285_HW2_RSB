"""Tiny CPU-only invariants: no network download, inference, or large matrix work."""

import importlib.util
import errno
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hw2_gan import evaluate_protocols as evaluation


def _cache_worker(directory, identity, counter, start, queue):
    cache = evaluation.ImmutableFeatureCache(directory)
    start.wait()

    def build():
        with counter.get_lock():
            counter.value += 1
        return {"features": np.arange(12, dtype=np.float32).reshape(4, 3)}

    arrays, hit = cache.get_or_create(identity, build, mode="create")
    queue.put((hit, evaluation.array_sha256(arrays["features"])))


class EvaluationSmoke(unittest.TestCase):
    def test_pixel_mmap_fallback_only_for_unsupported_mapping(self):
        pixels = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        for code in (errno.ENOSYS, errno.ENODEV, errno.EOPNOTSUPP):
            with patch.object(evaluation.np, "load", side_effect=[OSError(code, "unsupported"), pixels]) as loader:
                dataset = evaluation._ReferencePixels("/tmp/unopened", [0, 1])
                self.assertIs(dataset.images, pixels)
                self.assertIn(f"errno{code}", dataset.loading_method)
                self.assertEqual(loader.call_count, 2)
        with patch.object(evaluation.np, "load", side_effect=OSError(errno.EACCES, "permission")) as loader:
            with self.assertRaises(OSError):
                evaluation._ReferencePixels("/tmp/unopened", [0, 1])
            self.assertEqual(loader.call_count, 1)

    def test_atomic_immutable_cache_and_process_race(self):
        identity = {"protocol_id": "tiny-test-v1", "raw": "r", "preprocess": "p", "splits": "s", "indices": [2, 1]}
        context = mp.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            counter, start, queue = context.Value("i", 0), context.Event(), context.Queue()
            processes = [context.Process(target=_cache_worker, args=(directory, identity, counter, start, queue)) for _ in range(4)]
            for process in processes:
                process.start()
            start.set()
            results = [queue.get(timeout=15) for _ in processes]
            for process in processes:
                process.join(15)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(counter.value, 1)
            self.assertEqual(sum(not hit for hit, _ in results), 1)
            self.assertEqual(len({checksum for _, checksum in results}), 1)
            cache = evaluation.ImmutableFeatureCache(directory)
            arrays, hit = cache.get_or_create(identity, mode="readonly")
            self.assertTrue(hit)
            self.assertFalse(arrays["features"].flags.writeable)
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])
            for key in ("raw", "preprocess", "splits", "indices", "protocol_id"):
                changed = {**identity, key: [1, 2] if key == "indices" else "changed"}
                self.assertNotEqual(cache.path(identity), cache.path(changed))
                with self.assertRaises(FileNotFoundError):
                    cache.get_or_create(changed, mode="readonly")

    def test_corrupt_cache_is_rejected_and_preserved(self):
        identity = {"protocol_id": "tiny-test-v1", "indices": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            cache = evaluation.ImmutableFeatureCache(directory)
            cache.get_or_create(identity, lambda: {"features": np.ones((2, 3), dtype=np.float32)}, mode="create")
            path = cache.path(identity)
            with np.load(path, allow_pickle=False) as saved:
                metadata = saved["metadata"].copy()
            with path.open("wb") as handle:
                np.savez(handle, metadata=metadata, features=np.zeros((2, 3), dtype=np.float32))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "integrity"):
                cache.get_or_create(identity, lambda: {"features": np.ones((2, 3))}, mode="create")
            self.assertEqual(path.read_bytes(), before)

    def test_fixed_references_and_raw_content_exclusion(self):
        manifest = {"content_sha256": ["a", "b", "a", "c", "d", "b"]}
        self.assertEqual(evaluation._unique_content_indices([0, 1, 2, 3, 4, 5], manifest), [0, 1, 3, 4])
        self.assertEqual(evaluation._unique_content_indices([0, 1, 2, 3, 4, 5], manifest, excluded_hashes={"a", "b"}), [3, 4])
        before = torch.get_rng_state().clone()
        selected = evaluation._permuted(list(range(17)), 100043)
        expected = torch.randperm(17, generator=torch.Generator().manual_seed(100043)).tolist()
        self.assertEqual(selected, expected)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_inference_context_restores_rng_and_backend_on_failure(self):
        before_rng = torch.get_rng_state().clone()
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                  torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with evaluation._inference_context("cpu"):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.deterministic)
                torch.rand(3)
                raise RuntimeError("intentional test failure")
        after = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        self.assertEqual(before, after)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_exact_installed_torchvision_transform_algebra(self):
        from torchvision.models.inception import Inception3
        # Exercise only the installed algebra; no Inception construction/inference.
        holder = type("TransformOnly", (), {"transform_input": True})()
        rgb = torch.tensor([0, 0.2, 0.5, 1], dtype=torch.float64).reshape(1, 1, 2, 2).expand(1, 3, 2, 2)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb.dtype).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=rgb.dtype).reshape(1, 3, 1, 1)
        actual = Inception3._transform_input(holder, (rgb-mean)/std)
        torch.testing.assert_close(actual, 2*rgb-1, rtol=0, atol=5e-16)

    def test_tiny_fid_identities_against_independent_scipy(self):
        from scipy.linalg import sqrtm
        real = np.array([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
        moments = evaluation.moment_bundle(real)
        self.assertAlmostEqual(evaluation.pooled_fid(moments, real)["fid"], 0., places=12)
        translated = real + np.array([2., -3.])
        self.assertAlmostEqual(evaluation.pooled_fid(moments, translated)["fid"], 13., places=12)
        generated = real @ np.array([[1.3, 0.2], [-0.1, 0.7]]) + [0.4, -0.3]
        mu_r, mu_g = real.mean(0), generated.mean(0)
        cov_r, cov_g = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
        independent = np.sum((mu_r-mu_g)**2)+np.trace(cov_r+cov_g)-2*np.trace(sqrtm(cov_r@cov_g)).real
        self.assertAlmostEqual(evaluation.pooled_fid(moments, generated)["fid"], independent, places=12)

    def test_full_sample_is(self):
        self.assertAlmostEqual(evaluation.aggregate_inception_score(np.ones((4, 2))/2), 1.)
        self.assertAlmostEqual(evaluation.aggregate_inception_score(np.eye(2).repeat(2, axis=0)), 2.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
