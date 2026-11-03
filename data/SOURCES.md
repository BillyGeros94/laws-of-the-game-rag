# Sources

Provenance and metadata for every document in `data/raw/`.

- **Publisher:** The International Football Association Board (The IFAB)
- **Language:** English
- **Listing page:** https://www.theifab.com/documents/?documentType=all&language=en&years=2026
- **Retrieved:** 2026-10-23

## Included in the corpus

| filename | title | doc_type | date | status | reference |
|---|---|---|---|---|---|
| `laws_of_the_game_2026_27.pdf` | Laws of the Game 2026/27 (single-page edition) | laws | 2026/27 edition, effective 2026-07-01 | in_force | none |
| `circular_32_agm_decisions.pdf` | Circular no. 32: 140th Annual General Meeting of The IFAB, Decisions | circular | 2026-03-31 | in_force | SEC/2026-C483/bru-dre |
| `circular_33_additional_changes.pdf` | Circular no. 33: Additional changes to the Laws of the Game 2026/27 | circular | 2026-05-19 | in_force | SEC/2026-C486/bru |
| `circular_34_clarifications_and_trials.pdf` | Circular no. 34: Important clarifications to the Laws of the Game 2026/27 and information on trials relating to tactical goalkeeper injuries and the VAR protocol | circular | 2026-07-27 | in_force | SEC/2026-C487/bru |
| `trial_goalkeeper_injury.pdf` | Trial Protocol: Goalkeeper injury | trial | undated (enclosed with Circular 34, 2026-07-27) | trial | none |
| `trial_var_attacking_offence_set_pieces.pdf` | VAR Protocol Trial: Attacking-team offence before the ball is in play at attacking free kicks and corner kicks | trial | 2026-07 | trial | none |

Direct download links:

- `laws_of_the_game_2026_27.pdf`: <https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en>
- `circular_32_agm_decisions.pdf`: <https://downloads.theifab.com/downloads/circular-32?l=en>
- `circular_33_additional_changes.pdf`: <https://downloads.theifab.com/downloads/circular-33?l=en>
- `circular_34_clarifications_and_trials.pdf`: <https://downloads.theifab.com/downloads/circular-34?l=en>
- `trial_goalkeeper_injury.pdf`: <https://downloads.theifab.com/downloads/goalkeeper-injury?l=en>
- `trial_var_attacking_offence_set_pieces.pdf`: <https://downloads.theifab.com/downloads/attacking-team-offence-before-the-ball-is-in-play-at-attacking-free-kicks-and-cor?l=en>

## Notes per document type

- **laws:** the authoritative text. Some provisions are marked "competition option" in the text itself.
- **circular:** dated announcements. Later circulars clarify earlier ones (Circular 34 clarifies Circulars 32 and 33). Circular 34 states that the Circular 33 changes are not in the physical book; whether they appear in the downloadable Laws PDF is to be verified.
- **trial:** not part of the Laws. Use requires written permission from The IFAB (stated in Circular 34). Must not be presented as binding.

## Excluded from the corpus

Kept locally under `data/excluded/`, not indexed and not in the repository.

| filename | title | reason |
|---|---|---|
| `law_changes_2026_27.pdf` | Law changes 2026/27 (updated July 2026) | Diff document. Deleted text is struck through, which text extraction cannot distinguish from current text, so repealed wording would be retrieved as current. Known limitation. |
| `protocol_throw_in_goal_kick_countdown.pdf` | Throw-in and goal-kick countdown protocol | Presumed duplicate of the Laws book's "Notes and modifications". To be confirmed with the phrase probe. |
| `protocol_off_field_treatment_assessment.pdf` | Off-field treatment and assessment protocol | Same as above. |
| `protocol_time_limited_substitution.pdf` | Time-limited substitution protocol | Same as above. Its page footer is mislabelled "Throw-in and goal-kick countdown protocol" in the original file. |

## Conventions

- Citations use the filename plus the PDF page index (the page number a PDF viewer shows), never a page number parsed from the document text.
- Document titles in the index come from this file, never from page headers or footers.
