# Source attribution

Mesh retrieval and the accompanying object/metric utilities are adapted from
[Singapo](https://github.com/3dlg-hcvc/singapo), copyright (c) 2024 3dlg-hcvc,
under the [MIT license](singapo/LICENSE-SINGAPO).

This adaptation uses PWM geometry predictions and a prepared database in which
handles have been merged into their parent parts. Handles are not retrieved as
separate nodes. Geometry postprocessing and prediction assembly originate from
the PWM pipeline. Imports, file naming, orchestration, and error handling have
been adapted for standalone use.

Database preparation follows the PWM annotation conversion rules. It keeps
geometry and joint parameters unchanged, merges removed parts' mesh references
into retained parents, and rebuilds the topology hash after renumbering.
Non-Dishwasher shelves become doors; prismatic handles become drawers.

`assets/singapo_retrieval_reference.json` preserves the original Singapo candidate
catalog from the local `retrieval_hash_no_handles.json`, corresponding to
[Singapo's reference index](https://github.com/3dlg-hcvc/singapo/blob/main/retrieval/retrieval_hash_no_handles.json).
Only category/object membership is used; its original hash keys are not reused.
The resulting PWM index includes only successfully converted candidates whose
mesh references exist. Conversion and query hashing share the same implementation
and use NetworkX 3.4.2 as pinned in the retrieval section of `../requirements.txt`.
