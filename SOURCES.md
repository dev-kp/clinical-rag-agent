# Corpus sources

Three publisher-distinct guideline sets, chosen because they are (a) long and deeply
sectioned, (b) genuinely cross-referencing across chapters (multi-hop questions are
real, not manufactured), and (c) unambiguously reusable.

**Licence basis**

- **CDC** (`cdc.gov`): works of the U.S. federal government are public domain under
  17 U.S.C. §105 — no permission needed, attribution given as good practice.
- **Health Canada / PHAC** (`canada.ca`): Government of Canada web content is released
  under the [Open Government Licence – Canada](https://www.canada.ca/en/transparency/terms.html),
  which permits reproduction with attribution.

## Set A — CDC Sexually Transmitted Infections Treatment Guidelines, 2021

The syphilis cluster below is the corpus's main multi-hop engine: the treatment
regimen, the pregnancy-specific modification, and the penicillin-allergy workaround
live in three different chapters that all have to be combined to answer a realistic
question correctly.

| Doc ID | Title | URL |
|---|---|---|
| sti-intro | Summary and Introduction | https://www.cdc.gov/std/treatment-guidelines/intro.htm |
| sti-syphilis | Syphilis | https://www.cdc.gov/std/treatment-guidelines/syphilis.htm |
| sti-syphilis-primary-secondary | Primary and Secondary Syphilis | https://www.cdc.gov/std/treatment-guidelines/p-and-s-syphilis.htm |
| sti-syphilis-latent | Latent Syphilis | https://www.cdc.gov/std/treatment-guidelines/latent-syphilis.htm |
| sti-syphilis-neuro | Neurosyphilis, Ocular Syphilis, and Otosyphilis | https://www.cdc.gov/std/treatment-guidelines/neurosyphilis.htm |
| sti-syphilis-hiv | Syphilis Among Persons with HIV Infection | https://www.cdc.gov/std/treatment-guidelines/syphilis-hiv.htm |
| sti-syphilis-pregnancy | Syphilis During Pregnancy | https://www.cdc.gov/std/treatment-guidelines/syphilis-pregnancy.htm |
| sti-congenital-syphilis | Congenital Syphilis | https://www.cdc.gov/std/treatment-guidelines/congenital-syphilis.htm |
| sti-penicillin-allergy | Managing Persons with Penicillin Allergy History | https://www.cdc.gov/std/treatment-guidelines/penicillin-allergy.htm |
| sti-gonorrhea | Gonococcal Infections | https://www.cdc.gov/std/treatment-guidelines/gonorrhea.htm |
| sti-gonorrhea-adults | Gonococcal Infections Among Adolescents and Adults | https://www.cdc.gov/std/treatment-guidelines/gonorrhea-adults.htm |
| sti-chlamydia | Chlamydial Infections | https://www.cdc.gov/std/treatment-guidelines/chlamydia.htm |
| sti-pid | Pelvic Inflammatory Disease (PID) | https://www.cdc.gov/std/treatment-guidelines/pid.htm |

## Set B — CDC Clinical Practice Guideline for Prescribing Opioids for Pain, 2022

One long single-page document (MMWR Recommendations and Reports) covering 12
recommendations plus implementation considerations for renal/hepatic impairment,
pregnancy, and older adults — the exact "regimen in one section, dose adjustment in
another" pattern the plan calls out.

| Doc ID | Title | URL |
|---|---|---|
| opioid-rx-2022 | CDC Clinical Practice Guideline for Prescribing Opioids for Pain — United States, 2022 | https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm |

## Set C — Canadian Immunization Guide (Public Health Agency of Canada)

Second publisher/jurisdiction. Population-specific guidance (Part 3) cross-references
vaccine-specific chapters (Part 4) for dosing and contraindications — e.g. an
immunocompromised-host question needs both the population chapter and the specific
vaccine chapter.

| Doc ID | Title | URL |
|---|---|---|
| cig-schedules | Recommended immunization schedules (Part 1) | https://www.canada.ca/en/public-health/services/publications/healthy-living/canadian-immunization-guide-part-1-key-immunization-information/page-13-recommended-immunization-schedules.html |
| cig-populations | Vaccination of specific populations (Part 3, overview) | https://www.canada.ca/en/public-health/services/publications/healthy-living/canadian-immunization-guide-part-3-vaccination-specific-populations.html |
| cig-adults | Immunization of adults (Part 3) | https://www.canada.ca/en/public-health/services/publications/healthy-living/canadian-immunization-guide-part-3-vaccination-specific-populations/page-2-immunization-of-adults.html |
| cig-influenza | Influenza vaccines (Part 4) | https://www.canada.ca/en/public-health/services/publications/healthy-living/canadian-immunization-guide-part-4-active-vaccines/page-10-influenza-vaccine.html |
| cig-hepb | Hepatitis B vaccines (Part 4) | https://www.canada.ca/en/public-health/services/publications/healthy-living/canadian-immunization-guide-part-4-active-vaccines/page-7-hepatitis-b-vaccine.html |

**Retrieved**: fetch dates are recorded per-document in `ingest/raw/manifest.json` at
ingestion time, not hand-maintained here.

**Not medical advice**: this corpus and the system built on it are for a portfolio
demo only. Answers must never be presented as, or used as, clinical guidance for real
patients.
