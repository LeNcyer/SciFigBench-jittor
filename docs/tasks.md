# Task definitions

All boxes are [x1, y1, x2, y2] in original-image pixel coordinates. Images passed
to the model must be rendered consistently with the supplied QA.

| Task identifier | Question | Visual reference | Primary score |
|---|---|---|---|
| dense_content_perception | Transcribe boxed text | One red box, or multiple colors | ANLS |
| edge_level_verification | Is a direct edge allowed from source to target? | Red source, blue target | Accuracy |
| node_level_degree_counting | Count incoming/outgoing edges | Red target | Accuracy |
| path_level_reachability | Is a path of at least one allowed edge possible? | Red source, blue target | Accuracy |
| information_flow_tracing | List direct incoming/outgoing neighbor texts | Unique node label in prompt; no box | set-ANLS* |

Directed edges allow tail-to-head travel. Edges with two arrowheads or no
arrowhead are treated as bidirectional. A self-loop counts as one incoming and
one outgoing edge. Degree questions count parallel edges separately.

Information-flow questions count distinct neighbor nodes, preserving repeated
text when different nodes have identical labels. Multiple edges to the same
neighbor do not duplicate that neighbor. Answers may be empty.

Dense content uses red, orange, yellow, green, blue, violet order, skipping absent
colors. The single subtask has one red box. The flow_path subtask binds each
answer position to the corresponding color. Transcribe formulas as LaTeX and
replace line breaks within a label by a single space.

The renderer expands box borders outward using thickness
max(5, round(min(width, height) / 130)), clipped at image boundaries. It draws no
answers or diagnostic metadata. Information-flow images are returned as RGB
without drawing annotations.

The toolkit scores supplied ground truth; it does not construct the graph or
verify that user-supplied answers describe the image correctly.
