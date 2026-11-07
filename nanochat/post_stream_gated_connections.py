"""Post-mix counterpart with exactly the same parameters and gate inputs."""
from nanochat.stream_gated_connections import StreamGatedHyperConnection
from nanochat.post_stream_gate_mixing import post_stream_gate_mix


class PostStreamGatedHyperConnection(StreamGatedHyperConnection):
    def write(self,output,residual,mixes):
        post,h,buffer=mixes
        assert buffer is None
        return post_stream_gate_mix(residual,output.contiguous(),post.contiguous(),h.contiguous(),self.gate_weight)
