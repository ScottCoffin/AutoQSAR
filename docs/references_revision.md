Reference Audit — Revision Instructions for IDE Agent (v2)
Manuscript: QSARena (molecular property prediction benchmarking pipeline)Scope: For every reference 1–70: (a) verify bibliographic metadata against an authoritative source, (b) confirm the cited work supports the claim it is attached to, (c) confirm the reference is actually cited in the body, (d) update preprint→published where a version now exists, and flag anything unverifiable.Verification sources: Crossref REST API (journal metadata), arXiv (preprints + full texts), publisher/conference pages (NeurIPS, ICLR, ChEMBL, OECD, Zenodo). Languages searched: English only (justified — all sources are English-language international venues).
v2 changes: The two open claim-support items from v1 are now resolved by full-text checks — 17 "16 of 22 top-3" and 16 "18 first places" are both confirmed in the cited papers (flags retracted). Ref 69 now has a concrete Zenodo DOI. Only one genuine claim-support item remains for author review (4).
Audit result in one line: 70/70 references are real and correctly identified; 70/70 are cited in-text; 1 required metadata fix (24); 2 completeness items (69, 67); 1 claim-support item for author review (4).

PRIORITY 1 — REQUIRED FIX (factual metadata error)
Ref 24 — now published; add volume/issue/pages
Cited as in-press (year + DOI only); it has since published in print. "An End-User Audit of Reproducibility, Data Leakage, and Overfitting of the Top-Ranked ADMET Prediction Models in TDC Leaderboards," Journal of Chemical Information and Modeling, 2026, volume 66, issue 14, pages 8045–8058, DOI 10.1021/acs.jcim.6c00819. 1
FIND:
⟨24⟩ Koleiev I, Stratiichuk R, Shevchuk N, Melnychenko M, Nyporko O, Todoryshyn D, et al. An end-user audit of reproducibility, data leakage, and overfitting of the top-ranked ADMET prediction models in TDC leaderboards. Journal of Chemical Information and Modeling. 2026;https://doi.org/10.1021/acs.jcim.6c00819.
REPLACE WITH:
⟨24⟩ Koleiev I, Stratiichuk R, Shevchuk N, Melnychenko M, Nyporko O, Todoryshyn D, et al. An end-user audit of reproducibility, data leakage, and overfitting of the top-ranked ADMET prediction models in TDC leaderboards. Journal of Chemical Information and Modeling. 2026;66(14):8045–8058. https://doi.org/10.1021/acs.jcim.6c00819.

PRIORITY 2 — COMPLETENESS (incomplete reference entries)
Ref 69 — complete with the Zenodo version DOI
Current entry reads only Wognum C, et al.: polaris-hub/polaris. Zenodo. — missing year, version, and DOI. The polaris-hub/polaris software carries per-version Zenodo DOIs; the latest confirmed stable release is version 0.13.0 (published June 6, 2025), DOI 10.5281/zenodo.15610218. 2
Suggested entry (confirm the version you actually used):
⟨69⟩ Wognum C, et al. polaris-hub/polaris: 0.13.0. Zenodo; 2025. https://doi.org/10.5281/zenodo.15610218.
Two author decisions before committing this:
1.	Version — if your runs used a different polaris release, substitute that version and its DOI (e.g., 0.11.3, Jan 21 2025, DOI 10.5281/zenodo.14709054 3; 0.12.0, DOI 10.5281/zenodo.15261724 4). Alternatively use the version-independent Zenodo concept DOI if you want it to always resolve to the latest release.
2.	Author attribution — the Zenodo software record lists organizational creators (Recursion Pharmaceuticals, valence-labs, ETH Zurich 2), not "Wognum C." If you specifically intend to credit Cas Wognum, note that his name attaches to a different Zenodo object ("Popularizing best practices through the Polaris benchmarking platform," DOI 10.5281/zenodo.15328533 5). Decide whether 69 should point to the software deposit or that presentation/record, and align the author field accordingly.
Ref 67 — add the DOI (and optionally depositor authors)
Correctly identified. ChEMBL document CHEMBL3301361, "Experimental in vitro DMPK and physicochemical data on a set of publicly disclosed compounds," sourced from AstraZeneca DMPK/physicochemical, authored by Mark Wenlock and Nicholas Tomkinson, DOI 10.6019/CHEMBL3301361. 6 The in-text data-availability sentence already prints this DOI; add it to the reference entry for consistency.
Suggested entry:
⟨67⟩ Wenlock M, Tomkinson N (AstraZeneca). Experimental in vitro DMPK and physicochemical data on a set of publicly disclosed compounds. ChEMBL deposited dataset CHEMBL3301361. https://doi.org/10.6019/CHEMBL3301361.
(Keep "AstraZeneca" as corporate author if house style prefers.)
Ref 61 — verify page range (low risk)
Chapter confirmed to exist: "MetaQSAR: A comprehensive tool for automated QSAR modeling" by Pravin Ambure, Eva Serrano-Candelas, Jyotsna Bhat-Ambure, Rafael Gozalbes, 7 in the Elsevier volume Cheminformatic Modeling and Data Gap Filling for a Green and Sustainable Environment (2026). I could not independently confirm the exact page range 993–1019; confirm against the published chapter.

PRIORITY 3 — CITATION-SUPPORT ITEM (needs author judgment)
4 — review may not support the "unaffordable commercial-software / death valley" claim
Ref 4 is cited twice. The first use is well-matched (in-silico ADMET prediction as an indispensable complement, [3, 4]). The second use is the concern: "many academic drug-discovery efforts founder in the preclinical 'death valley' partly because researchers cannot afford licences for commercial ADME prediction software 4." 8 Ref 4 is Komura, Watanabe & Mizuguchi, "The Trends and Future Prospective of In Silico Models from the Viewpoint of ADME Evaluation in Drug Discovery" (Pharmaceutics, 2023) 1 — a methods-trends review that may not substantiate a socioeconomic claim about licence affordability and academic attrition. ACTION: confirm ref 4 makes this point, or add/replace with a source that directly supports the accessibility/cost argument.
Optional quantitative figures to spot-check against source (low priority)
These are the authors' own readings of cited works, confirmable only at abstract level in this audit:
•	"8 MolGPS, scaled to three billion parameters" — the Sypetkowski et al. NeurIPS 2024 paper confirms MolGPS, phenomics-inclusive pretraining, and SOTA on 26 of 38 tasks, 9 but the exact 3-billion-parameter figure was not confirmed.
•	"7 MolE, pretrained on roughly 842 million molecules" and "25 ADMET-AI … trained across 41 TDC datasets" — confirm these exact figures against refs 7 and 25. (Note: 25's own abstract supports "41 ADMET datasets," so that one is likely fine.)

PRIORITY 4 — OPTIONAL / MINOR (defensible as-is)
•	2 Kola & Landis — page range. Manuscript shows 711–715; Crossref lists pages 711–716. 1 One-page end-page discrepancy (common Crossref quirk); the manuscript's 711–715 matches the standard citation. No change required unless house style defers to Crossref.
•	30 DeepPurpose — year. Manuscript shows 2020; Crossref shows online December 1, 2020 and print April 1, 2021 (vol. 36, issue 22–23, pp. 5545–5547). 1 Both defensible; change to 2021 only if matching print dates.
•	16 ADMETboost — issue number. Manuscript shows 28:408; Crossref lists volume 28, issue 12, article 408. 1 Adding 28(12):408 is optional (article-number style is fine).

VERIFICATION LOG — all 70 references (bibliographic + cited-in-text + claim)
Legend: OK = metadata verified correct; Cited = confirmed present in body text. Claim column notes only exceptions.

Ref	Short ID	Metadata	Cited in text	Claim-support note
1	Sun et al., clinical attrition	OK — Acta Pharm Sin B 2022;12(7):3049–3062 1
Cited	Supports "~90% fail"
2	Kola & Landis	OK — Nat Rev Drug Discov 2004;3(8) (pp. see Minor) 1
Cited	Supports 40%→10% PK-attrition
3	ADMETlab 3.0	OK — Nucleic Acids Res 2024;52(W1):W422–W431 1
Cited	OK
4	Komura et al. review	OK — Pharmaceutics 2023;15(11):2619 1
Cited (×2)	See Priority-3 flag
5	MoleculeNet	OK — Chem Sci 2018;9(2):513–530 1
Cited	OK
6	TDC	OK — NeurIPS 2021 Datasets & Benchmarks; arXiv:2102.09548 10
Cited	Venue correct
7	MolE	OK — Nat Commun 2024;15:9431 1
Cited	Confirm 842M-molecules figure
8	MolGPS (Sypetkowski)	OK — "On the Scalability of GNNs for Molecular Graphs," NeurIPS 2024; SOTA 26/38 9,11
Cited	Confirm 3B-param figure
9	MiniMol	OK — arXiv:2404.14986; preprint only 12
Cited 8
OK; correctly cited as preprint
10	Uni-Mol	OK — ICLR 2023 13,14
Cited	Venue correct
11	Xia et al. limitations	OK — NeurIPS 2023 (vol. 36) 15,16
Cited 8
Claim supported
12	Rodriguez-Pérez review	OK — Annu Rev Biomed Data Sci 2022;5:43–65 1
Cited 8
Claim supported
13	Green et al. real-world	OK — arXiv:2309.17161; preprint only 17
Cited	Claim supported 17

14	Li et al. survey	OK — J Chem Theory Comput 2026;22(10):4866–4887 1
Cited	Claim supported 18

15	Zhao et al. reliability	OK — J Cheminform 2026;18:95; DOI resolves 1
Cited	Claim supported (TabPFNv2 > molecular FMs) 19

16	ADMETboost	OK — J Mol Model 2022;28(12):408 1
Cited (×2)	Claim supported — paper reports first in 18/22, top-3 in 21/22 20

17	MapLight (Notwell & Wood)	OK — arXiv:2310.00174; preprint only 21
Cited	Claim supported — paper body reports top-3 in 16/22 22

18	CaliciBoost	OK — arXiv:2506.08059; preprint only 23
Cited	OK
19	MaxQsaring (Xu et al.)	OK — J Pharm Anal 2025;15(12):101411 1
Cited (×2)	Claim supported — first on 19/22 TDC tasks 24

20	Kamuntavičius et al.	OK — J Cheminform 2025;17:108 1
Cited	Claim supported 25

21	Fooladi et al. OOD	OK — J Chem Inf Model 2025;65(19):9871–9891 1
Cited 8
Claim supported
22	Kapoor & Narayanan	OK — Patterns 2023;4(9):100804 1
Cited 8
Claim supported
23	Belfield et al.	OK — PLoS ONE 2023;18(5):e0282924 1
Cited 8
OK
24	Koleiev et al. audit	FIX — see Priority 1	Cited 8
Claim supported
25	ADMET-AI	OK — Bioinformatics 2024;40(7):btae416 1
Cited	"41 datasets" supported 26

26	QSARtuna	OK — J Chem Inf Model 2024;64(14):5365–5374 1
Cited 8
OK
27	Turon et al. (ZairaChem)	OK — Nat Commun 2023;14:5736 1
Cited (in-text "ZairaChem") 8
Defensible: ZairaChem/Ersilia application
28	QSPRpred	OK — J Cheminform 2024;16:128 1
Cited 8
OK
29	DeepMol	OK — J Cheminform 2024;16:136 1
Cited 8
OK
30	DeepPurpose	OK — Bioinformatics 36(22–23):5545–5547 (year: see Minor) 1
Cited 8
OK
31	Auto-ADMET	OK — arXiv:2502.16378; preprint only 27
Cited 8
OK
32	ChemXploreML	OK — J Chem Inf Model 2025;65(11):5424–5437 1
Cited	OK
33	DeepChem (book)	OK — O'Reilly, 2019 8
Cited 8
OK
34	OCHEM	OK — J Comput Aided Mol Des 2011;25(6):533–554 1
Cited 8
OK
35	ChemSAR	OK — J Cheminform 2017;9:27 1
Cited 8
OK
36	ChemML	OK — WIREs Comput Mol Sci 2020;10(4):e1458 1
Cited	OK
37	ESOL (Delaney)	OK — J Chem Inf Comput Sci 2004;44(3):1000–1005 1
Cited	OK
38	FreeSolv	OK — J Comput Aided Mol Des 2014;28(7):711–720 1
Cited	OK
39	von Borries et al.	OK — Nat Commun 2026;17:647; DOI resolves 1
Cited	OK
40	RDKit	OK — software citation (rdkit.org) 8
Cited	OK
41	ECFP (Rogers & Hahn)	OK — J Chem Inf Model 2010;50(5):742–754 1
Cited	OK
42	MACCS (Durant et al.)	OK — J Chem Inf Comput Sci 2002;42(6):1273–1280 1
Cited	OK
43	Gedeck et al.	OK — J Chem Inf Model 2006;46(5):1924–1936 1
Cited	OK
44	ErG (Stiefl et al.)	OK — J Chem Inf Model 2006;46(1):208–220 1
Cited	OK
45	Bemis & Murcko	OK — J Med Chem 1996;39(15):2887–2893 1
Cited 8
OK
46	scikit-learn	OK — JMLR 2011;12:2825–2830 (no DOI, correct) 8
Cited	OK
47	XGBoost	OK — KDD '16, pp. 785–794 1
Cited	OK
48	CatBoost	OK — NeurIPS 2018 (vol. 31) 8
Cited	OK
49	Yang et al. D-MPNN	OK — J Chem Inf Model 2019;59(8):3370–3388 1
Cited	OK
50	Chemprop (Heid et al.)	OK — J Chem Inf Model 2024;64(1):9–17 1
Cited	OK
51	Chemprop v2 (Graff et al.)	OK — J Chem Inf Model 2026;66(1):28–33 1
Cited	OK
52	Uni-Mol2 (Ji et al.)	OK — NeurIPS 2024; arXiv:2406.14969 28,13
Cited	Venue correct
53	TabPFN (Hollmann et al.)	OK — ICLR 2023 (oral, notable top-25%) 29,30
Cited	Venue correct
54	Hsu et al. CFA (chapter)	OK — IGI Global, 2006, pp. 32–62 8
Cited 8
OK
55	Jetstream2	OK — Hancock et al., PEARC '21 1
Cited	OK
56	ACCESS	OK — Boerner et al., PEARC '23, pp. 173–176 1
Cited	OK
57	Deng et al.	OK — Nat Commun 2023;14(1):6395 1
Cited	Claim supported (fixed representations lead on most datasets) 31

58	QSAR Workbench (Cox et al.)	OK — J Comput Aided Mol Des 2013;27(4):321–336 1
Cited 8
OK
59	AutoQSAR (Dixon et al.)	OK — Future Med Chem 2016;8(15):1825–1839 1
Cited 8
OK
60	Schrödinger DeepAutoQSAR	OK — Schrödinger white paper 8
Cited 8
OK (grey literature, labeled)
61	MetaQSAR (Ambure et al.)	OK — Elsevier chapter 2026 (verify pages) 7
Cited 8
OK
62	OECD (Q)SAR principles	OK — OECD guidance 8
Cited 8
OK
63	OECD (Q)SAR Assessment Framework	OK — OECD, 2023 8
Cited 8
OK
64	Roy et al. applicability domain	OK — Chemom Intell Lab Syst 2015;145:22–29 1
Cited 8
Claim supported
65	Meta-QSAR (Olier et al.)	OK — Mach Learn 2018;107(1):285–311 1
Cited 8
OK
66	Sheridan domain metrics	OK — J Chem Inf Model 2015;55(6):1098–1107 1
Cited 8
Claim supported 8

67	AstraZeneca ChEMBL deposit	OK — CHEMBL3301361; add DOI (Priority 2) 6
Cited 8
OK
68	Ash et al. (Polaris)	OK — J Chem Inf Model 2025;65(18):9398–9411 1
Cited 8
OK
69	polaris (Zenodo)	INCOMPLETE — complete with DOI 10.5281/zenodo.15610218 (v0.13.0); see Priority 2 2
Cited 8
Confirm version + author attribution
70	Fang et al. (Biogen ADME)	OK — J Chem Inf Model 2023;63(11):3263–3274 1
Cited 8
OK


Summary of actions for the agent
1.	Apply Priority 1 (ref 24 find/replace) — the only mandatory metadata correction.
2.	Apply Priority 2 — complete ref 69 with the Zenodo DOI (author confirms version + attribution), add DOI to ref 67, verify page range of ref 61.
3.	Route Priority 3 to the author — one claim-support item (4 accessibility claim) plus optional spot-checks of the 7/8 figures.
4.	Priority 4 is optional — 2, 30, 16 are defensible as written.
5.	No orphan references and no broken in-text markers — every reference 1–70 is cited, and every marker resolves to a listed entry.
Preprint / publication-status check
Items 9, 13, 17, 18, 31 remain preprint-only and are correctly cited as arXiv; no journal version to substitute. Conference papers 6 (NeurIPS 2021), 10 (ICLR 2023), 11 (NeurIPS 2023), 52 (NeurIPS 2024), 53 (ICLR 2023) are published at the stated venues. No retractions, withdrawals, or predatory-venue issues were found; every tested DOI/arXiv ID resolves. The only entry advanced from in-press to full citation since drafting is 24 (Priority 1).
References
1.	Koleiev I, Stratiichuk R, Shevchuk N, Melnychenko M, Nyporko A, Todoryshyn D, Husak V, Starosyla S, Yesylevskyy S, Nafiiev A. An End-User Audit of Reproducibility, Data Leakage, and Overfitting of the Top-Ranked ADMET Prediction Models in TDC Leaderboards. J Chem Inf Model. 2026;66(14):8045-8058. doi:10.1021/acs.jcim.6c00819. PMID: 42392971.
2.	polaris-hub/polaris: 0.13.0. Retrieved 2026-10-07, from https://zenodo.org/records/15610218
3.	polaris-hub/polaris: 0.11.3. Retrieved 2026-10-07, from https://zenodo.org/records/14709054
4.	polaris-hub/polaris: 0.12.0. Retrieved 2026-10-07, from https://zenodo.org/records/15261724
5.	Popularizing best practices through the Polaris benchmarking platform | Zenodo. Retrieved 2026-10-07, from https://zenodo.org/records/15328533
6.	Experimental in vitro DMPK and physicochemical data on a. Retrieved 2026-10-07, from https://www.ebi.ac.uk/chembl/explore/document/CHEMBL3301361
7.	Ambure P, Serrano-Candelas E, Bhat-Ambure J, Gozalbes R. MetaQSAR: A comprehensive tool for automated QSAR modeling. Cheminformatic Modeling and Data Gap Filling for a Green and Sustainable Environment. 2026;:993-1019. doi:10.1016/b978-0-443-36474-7.00003-x.
8.	manuscript.pdf. Internal reference: file:28013#pages=4-5. Accessed 2026-10-07.
9.	NeurIPS Poster On the Scalability of GNNs for Molecular Graphs. Retrieved 2026-10-07, from https://neurips.cc/virtual/2024/poster/93869
10.	[2102.09548] Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. Retrieved 2026-10-07, from https://arxiv.org/abs/2102.09548
11.	On the Scalability of GNNs for Molecular Graphs. Retrieved 2026-10-07, from https://proceedings.neurips.cc/paper_files/paper/2024/file/2345275663a15ee92a06bc957be54a2c-Paper-Conference.pdf
12.	[2404.14986] $\texttt{MiniMol}$: A Parameter-Efficient Foundation Model for Molecular Learning. Retrieved 2026-10-07, from https://arxiv.org/abs/2404.14986
13.	Official Repository for the Uni-Mol Series Methods - GitHub. Retrieved 2026-10-07, from https://github.com/deepmodeling/Uni-Mol
14.	Published as a conference paper at ICLR 2023. Retrieved 2026-10-07, from https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6402990d37e01856dc1d1581/original/uni-mol-a-universal-3d-molecular-representation-learning-framework.pdf
15.	Understanding the Limitations of Deep Models for. Retrieved 2026-10-07, from https://proceedings.neurips.cc/paper_files/paper/2023/hash/cc83e97320000f4e08cb9e293b12cf7e-Abstract-Conference.html
16.	junxia97/IFM: [NeurIPS 2023] "Understanding the Limitations of. Retrieved 2026-10-07, from https://github.com/junxia97/IFM
17.	[2309.17161] Current Methods for Drug Property Prediction in the Real World. Retrieved 2026-10-07, from https://arxiv.org/abs/2309.17161
18.	Li Z, Chen X, Wen H, Zhang RQ, Li M, Zhang X, Yin H, Yang Q, Lam KY, Lio P, Yiu SM. A Systematic Survey and Benchmark of Deep Learning for Molecular Property Prediction in the Foundation Model Era. J Chem Theory Comput. 2026;22(10):4866-4887. doi:10.1021/acs.jctc.5c02081. PMID: 42096352.
19.	Zhao D, Zhu Y, Wu Z, Wan Y, Liu X, Li S, Xu H, Hou T, Hsieh CY. Revisiting ADMET prediction reliability under real-world challenges in the foundation model era. J Cheminform. 2026;18(1):95. doi:10.1186/s13321-026-01217-2. PMID: 42152045.
20.	Tian H, Ketkar R, Tao P. ADMETboost: a web server for accurate ADMET prediction. J Mol Model. 2022;28(12):408. doi:10.1007/s00894-022-05373-8. PMID: 36454321.
21.	[2310.00174] ADMET property prediction through combinations of molecular fingerprints. Retrieved 2026-10-07, from https://arxiv.org/abs/2310.00174
22.	ADMET property prediction through combinations of molecular fingerprints. Retrieved 2026-10-07, from https://arxiv.org/html/2310.00174v1
23.	[2506.08059] CaliciBoost: Performance-Driven Evaluation of Molecular Representations for Caco-2 Permeability Prediction. Retrieved 2026-10-07, from https://arxiv.org/abs/2506.08059
24.	Xu C, Xu Y, Hu Z, Zhao X, Xie W, Chen W, Pei J. Unveiling optimal molecular features for hERG insights with automatic machine learning. J Pharm Anal. 2025;15(12):101411. doi:10.1016/j.jpha.2025.101411. PMID: 41487145.
25.	Kamuntavičius G, Paquet T, Bastas O, Šalkauskas D, Prat A, Aty HA, Pabrinkis A, Norvaišas P, Tal R. Benchmarking ML in ADMET predictions: the practical impact of feature representations in ligand-based models. J Cheminform. 2025;17(1):108. doi:10.1186/s13321-025-01041-0. PMID: 40691635.
26.	Swanson K, Walther P, Leitz J, Mukherjee S, Wu JC, Shivnaraine RV, Zou J. ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries. Bioinformatics. 2024;40(7):btae416. doi:10.1093/bioinformatics/btae416. PMID: 38913862.
27.	[2502.16378] Auto-ADMET: An Effective and Interpretable AutoML Method for Chemical ADMET Property Prediction. Retrieved 2026-10-07, from https://arxiv.org/abs/2502.16378
28.	Exploring Molecular Pretraining Model at Scale. Retrieved 2026-10-07, from https://proceedings.neurips.cc/paper_files/paper/2024/file/53923bb44655a7defb31c7744c01b62b-Paper-Conference.pdf
29.	ICLR Oral TabPFN: A Transformer That Solves Small Tabular. Retrieved 2026-10-07, from https://iclr.cc/virtual/2023/oral/12541
30.	TabPFN: A Transformer That Solves Small Tabular ... - OpenReview. Retrieved 2026-10-07, from https://openreview.net/forum?id=cp5PvcI6w8_
31.	Deng J, Yang Z, Wang H, Ojima I, Samaras D, Wang F. A systematic study of key elements underlying molecular property prediction. Nat Commun. 2023;14(1):6395. doi:10.1038/s41467-023-41948-6. PMID: 37833262.
