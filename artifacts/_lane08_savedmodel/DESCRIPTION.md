# Lane SavedModel Export Description

This folder is a TensorFlow SavedModel export for the lane deployment model.

Contents:

- `saved_model.pb`: serialized graph and serving signatures.
- `fingerprint.pb`: model fingerprint metadata.
- `assets/`: optional runtime assets.
- `variables/`: tensor variable checkpoint files.

This export is typically an intermediate source for additional conversion targets (for example OpenVINO or TFLite workflows).
