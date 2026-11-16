# Tasks

| Task | Public answer format | Metric |
|---|---|---|
| dense_content_perception | List of strings in color order | Position-wise ANLS |
| edge_level_verification | String true/false | Accuracy |
| node_level_degree_counting | Nonnegative integer string | Accuracy |
| path_level_reachability | String true/false | Accuracy |
| information_flow_tracing | JSON-encoded list of strings | Set-based ANLS* |

Model responses for every task are raw strings, including JSON-encoded lists.
DCP binds boxes to red, orange, yellow, green, blue, violet in that order; single
questions use item 0. Edge asks for a direct directed connection. Degree counts
incoming/outgoing edges. Reachability follows directed paths. IFT identifies
direct incoming/outgoing neighbors from their text and preserves multiplicity.
Its canonical input is the original RGB image without a bounding box overlay.

All prompts and visual conventions follow the official toolkit. Only prompt and
canonical image are sent to the user model. See evaluation.md for partial coverage,
empty-answer and parsing semantics, and AGENTS.md for the complete API.
