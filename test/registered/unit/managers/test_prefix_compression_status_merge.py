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
                "payload_bytes_compressed": compressed * 40,
                "state_bytes_compressed": compressed * 10,
                "metadata_bytes_compressed": compressed * 2,
                "global_bytes": 20,
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
