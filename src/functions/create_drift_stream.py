import v3io.dataplane

def create_drift_stream(context, stream_path: str, container: str = "users"):
    v3io_client = v3io.dataplane.Client()
    v3io_client.stream.create(
        container=container,
        stream_path=stream_path,
        shard_count=1,
        raise_for_status=v3io.dataplane.RaiseForStatus.never
    )