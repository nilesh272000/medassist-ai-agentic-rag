# Architecture

Intent Router → Vector Retrieval → Evidence Grader → Grounded Generation

If evidence is weak: Query Rewrite → Retrieval Retry → Safe Fallback.

The API also exposes an image-observation endpoint for multimodal demonstration. The image endpoint is descriptive and explicitly avoids diagnosis.