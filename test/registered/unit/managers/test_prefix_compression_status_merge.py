from sglang.srt.managers.io_struct import PrefixCompressionReqOutput
from sglang.srt.managers.tokenizer_control_mixin import (
    _merge_prefix_compression_results,
)


def _status(stored, used, peak, compressed):
    return PrefixCompressionReqOutput(
        data={
            "idle": True,
            "stored_blocks": 0,
            "stored_requests": stored,
            "store": {
                "storage_unit": "request",
                "store_bytes": 1000,
                "free_bytes": 1000 - used,
                "stored_requests": stored,
                "evictions": 1,
                "used_bytes": used,
                "peak_used_bytes": peak,
                "requests_compressed": compressed,
                "original_bytes_compressed": compressed * 100,
                "exposed_bytes_compressed": compressed * 100,
                "exposed_kv_bytes_compressed": compressed * 90,
                "exposed_temporal_bytes_compressed": compressed * 10,
                "payload_bytes_compressed": compressed * 40,
                "kv_payload_bytes_compressed": compressed * 35,
                "temporal_payload_bytes_compressed": compressed * 5,
                "payload_attribution_complete": True,
                "state_bytes_compressed": compressed * 10,
                "metadata_bytes_compressed": compressed * 2,
                "global_bytes": 20,
                # One exemplar unit, the same on every rank.
                "sample_exposed_bytes": 100,
                "sample_payload_bytes": 40,
            },
        }
    )


def test_request_store_status_sums_residency_and_cumulative_counts_across_dp():
    merged = _merge_prefix_compression_results(
        "status", [_status(2, 200, 250, 3), _status(1, 100, 150, 2)]
    ).data

    assert merged["stored_requests"] == 3
    assert merged["store"]["stored_requests"] == 3
    assert merged["store"]["used_bytes"] == 300
    assert merged["store"]["peak_used_bytes"] == 400
    assert merged["store"]["requests_compressed"] == 5
    assert merged["store"]["global_bytes"] == 40
    assert merged["store"]["store_bytes"] == 2000
    assert merged["store"]["free_bytes"] == 1700
    assert merged["dp_size"] == 2


def test_request_store_status_sums_component_byte_counters_across_dp():
    """Exposed and payload totals must count the same ranks.

    The merge summed `payload_bytes_compressed` while the per-component and exposed
    counters kept rank 0's value, so a saving computed from the pair came back negative:
    a codec compressing 3.88x reported 0.97x on four ranks.
    """
    merged = _merge_prefix_compression_results(
        "status", [_status(2, 200, 250, 3), _status(1, 100, 150, 2)]
    ).data["store"]

    assert merged["exposed_bytes_compressed"] == 500
    assert merged["exposed_kv_bytes_compressed"] == 450
    assert merged["exposed_temporal_bytes_compressed"] == 50
    assert merged["payload_bytes_compressed"] == 200
    assert merged["kv_payload_bytes_compressed"] == 175
    assert merged["temporal_payload_bytes_compressed"] == 25
    # Per-unit exemplars describe one request, so they are not counts to add up.
    assert merged["sample_exposed_bytes"] == 100
    assert merged["sample_payload_bytes"] == 40
    assert merged["payload_attribution_complete"] is True


def test_request_store_attribution_is_incomplete_when_any_rank_cannot_attribute():
    partial = _status(1, 100, 150, 2)
    partial.data["store"]["payload_attribution_complete"] = False

    merged = _merge_prefix_compression_results(
        "status", [_status(2, 200, 250, 3), partial]
    ).data["store"]

    assert merged["payload_attribution_complete"] is False


def test_block_store_status_sums_cumulative_metadata_across_dp():
    def status(metadata):
        return PrefixCompressionReqOutput(
            data={
                "idle": True,
                "stored_blocks": 1,
                "stored_requests": 0,
                "store": {
                    "storage_unit": "block",
                    "stored_blocks": 1,
                    "evictions": 2,
                    "metadata_bytes_compressed": metadata,
                },
            }
        )

    merged = _merge_prefix_compression_results(
        "status", [status(11), status(17)]
    ).data

    assert merged["stored_blocks"] == 2
    assert merged["store"]["stored_blocks"] == 2
    assert merged["store"]["evictions"] == 4
    assert merged["store"]["metadata_bytes_compressed"] == 28
