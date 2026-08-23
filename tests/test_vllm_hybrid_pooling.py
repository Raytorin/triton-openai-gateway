import json
from pathlib import Path
import sys
import tempfile
import unittest


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backends" / "vllm_multimodal"
sys.path.insert(0, str(BACKEND_ROOT))

from utils.hybrid_embeddings import (  # noqa: E402
    PoolingModelMetadata,
    build_sparse_embedding,
    load_pooling_model_metadata,
    parse_embedding_output_spec,
    serialize_pooling_output,
)


class VllmHybridPoolingTests(unittest.TestCase):
    def setUp(self):
        self.metadata = PoolingModelMetadata(
            architecture="BgeM3EmbeddingModel",
            hidden_size=4,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
            unk_token_id=3,
        )

    def test_pooling_task_is_selected_from_output_types(self):
        dense = parse_embedding_output_spec({"pooling_params": {}})
        sparse = parse_embedding_output_spec(
            {"pooling_params": {}, "output_types": ["sparse"]}
        )
        hybrid = parse_embedding_output_spec(
            {
                "pooling_params": {"dimensions": [4]},
                "output_types": ["dense", "sparse"],
                "sparse_top_k": 2,
            }
        )

        self.assertEqual("embed", dense.task)
        self.assertFalse(dense.explicit_output_types)
        self.assertEqual("token_classify", sparse.task)
        self.assertEqual("embed&token_classify", hybrid.task)
        self.assertEqual(4, hybrid.dimensions)
        self.assertEqual(2, hybrid.sparse_top_k)

    def test_invalid_sparse_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sparse_top_k requires sparse"):
            parse_embedding_output_spec(
                {
                    "pooling_params": {},
                    "output_types": ["dense"],
                    "sparse_top_k": 4,
                }
            )

        with self.assertRaisesRegex(ValueError, "unsupported embedding output"):
            parse_embedding_output_spec(
                {"pooling_params": {}, "output_types": ["colbert"]}
            )

    def test_legacy_dense_response_remains_a_vector(self):
        spec = parse_embedding_output_spec({"pooling_params": {}})

        result = serialize_pooling_output([0.1, 0.2], [0, 2], spec, self.metadata)

        self.assertEqual([0.1, 0.2], result)

    def test_sparse_output_filters_special_tokens_and_deduplicates(self):
        sparse = build_sparse_embedding(
            [10, 10, 3, 20],
            [0.2, 0.8, 1.0, 0.4],
            self.metadata.special_token_ids,
            sparse_top_k=1,
        )

        self.assertEqual([10], sparse["indices"])
        self.assertEqual([0.8], sparse["values"])

    def test_hybrid_output_is_split_and_mapped_to_prompt_tokens(self):
        spec = parse_embedding_output_spec(
            {
                "pooling_params": {},
                "output_types": ["dense", "sparse"],
                "sparse_top_k": 2,
            }
        )

        result = serialize_pooling_output(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.7],
            [0, 10, 10, 20, 2],
            spec,
            self.metadata,
        )

        self.assertEqual([0.1, 0.2, 0.3, 0.4], result["dense"])
        self.assertEqual([10, 20], result["sparse"]["indices"])
        self.assertEqual([0.9, 0.7], result["sparse"]["values"])

    def test_sparse_only_output_is_mapped_to_prompt_tokens(self):
        spec = parse_embedding_output_spec(
            {"pooling_params": {}, "output_types": ["sparse"]}
        )

        result = serialize_pooling_output(
            [0.5, 0.9, 0.7],
            [0, 10, 10, 20, 2],
            spec,
            self.metadata,
        )

        self.assertEqual([10, 20], result["sparse"]["indices"])
        self.assertEqual([0.9, 0.7], result["sparse"]["values"])

    def test_sparse_output_length_mismatch_is_rejected(self):
        spec = parse_embedding_output_spec(
            {"pooling_params": {}, "output_types": ["sparse"]}
        )

        with self.assertRaisesRegex(ValueError, "does not match prompt tokens"):
            serialize_pooling_output(
                [0.5],
                [0, 10, 20, 2],
                spec,
                self.metadata,
            )

    def test_sparse_output_requires_bge_m3_architecture(self):
        spec = parse_embedding_output_spec(
            {"pooling_params": {}, "output_types": ["sparse"]}
        )

        with self.assertRaisesRegex(ValueError, "BgeM3EmbeddingModel"):
            serialize_pooling_output(
                [0.5],
                [10],
                spec,
                PoolingModelMetadata(architecture="XLMRobertaModel"),
            )

    def test_model_metadata_applies_hf_architecture_override(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["XLMRobertaModel"],
                        "hidden_size": 1024,
                        "bos_token_id": 0,
                        "eos_token_id": 2,
                    }
                ),
                encoding="utf-8",
            )
            (model_path / "tokenizer.json").write_text(
                json.dumps(
                    {
                        "added_tokens": [
                            {"id": 0, "content": "<s>", "special": True},
                            {"id": 3, "content": "<unk>", "special": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_pooling_model_metadata(
                {
                    "model": str(model_path),
                    "hf_overrides": {"architectures": ["BgeM3EmbeddingModel"]},
                }
            )

        self.assertTrue(metadata.supports_bge_m3_sparse)
        self.assertEqual(1024, metadata.hidden_size)
        self.assertEqual(0, metadata.bos_token_id)
        self.assertEqual(2, metadata.eos_token_id)
        self.assertEqual(frozenset({0, 3}), metadata.additional_special_token_ids)

    def test_non_bge_model_metadata_does_not_parse_token_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3ForCausalLM"],
                        "eos_token_id": [151643, 151645],
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_pooling_model_metadata({"model": str(model_path)})

        self.assertEqual("Qwen3ForCausalLM", metadata.architecture)
        self.assertIsNone(metadata.eos_token_id)


if __name__ == "__main__":
    unittest.main()
