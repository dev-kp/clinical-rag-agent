from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    url: str


# Mirrors SOURCES.md. Keep the two in sync when the corpus changes.
MANIFEST: list[SourceDoc] = [
    # Set A — CDC STI Treatment Guidelines, 2021
    SourceDoc("sti-intro", "Summary and Introduction", "https://www.cdc.gov/std/treatment-guidelines/intro.htm"),
    SourceDoc("sti-syphilis", "Syphilis", "https://www.cdc.gov/std/treatment-guidelines/syphilis.htm"),
    SourceDoc(
        "sti-syphilis-primary-secondary",
        "Primary and Secondary Syphilis",
        "https://www.cdc.gov/std/treatment-guidelines/p-and-s-syphilis.htm",
    ),
    SourceDoc(
        "sti-syphilis-latent",
        "Latent Syphilis",
        "https://www.cdc.gov/std/treatment-guidelines/latent-syphilis.htm",
    ),
    SourceDoc(
        "sti-syphilis-neuro",
        "Neurosyphilis, Ocular Syphilis, and Otosyphilis",
        "https://www.cdc.gov/std/treatment-guidelines/neurosyphilis.htm",
    ),
    SourceDoc(
        "sti-syphilis-hiv",
        "Syphilis Among Persons with HIV Infection",
        "https://www.cdc.gov/std/treatment-guidelines/syphilis-hiv.htm",
    ),
    SourceDoc(
        "sti-syphilis-pregnancy",
        "Syphilis During Pregnancy",
        "https://www.cdc.gov/std/treatment-guidelines/syphilis-pregnancy.htm",
    ),
    SourceDoc(
        "sti-congenital-syphilis",
        "Congenital Syphilis",
        "https://www.cdc.gov/std/treatment-guidelines/congenital-syphilis.htm",
    ),
    SourceDoc(
        "sti-penicillin-allergy",
        "Managing Persons with Penicillin Allergy History",
        "https://www.cdc.gov/std/treatment-guidelines/penicillin-allergy.htm",
    ),
    SourceDoc("sti-gonorrhea", "Gonococcal Infections", "https://www.cdc.gov/std/treatment-guidelines/gonorrhea.htm"),
    SourceDoc(
        "sti-gonorrhea-adults",
        "Gonococcal Infections Among Adolescents and Adults",
        "https://www.cdc.gov/std/treatment-guidelines/gonorrhea-adults.htm",
    ),
    SourceDoc("sti-chlamydia", "Chlamydial Infections", "https://www.cdc.gov/std/treatment-guidelines/chlamydia.htm"),
    SourceDoc("sti-pid", "Pelvic Inflammatory Disease (PID)", "https://www.cdc.gov/std/treatment-guidelines/pid.htm"),
    # Set B — CDC Clinical Practice Guideline for Prescribing Opioids for Pain, 2022
    SourceDoc(
        "opioid-rx-2022",
        "CDC Clinical Practice Guideline for Prescribing Opioids for Pain — United States, 2022",
        "https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm",
    ),
    # Set C — Canadian Immunization Guide (Public Health Agency of Canada)
    SourceDoc(
        "cig-schedules",
        "Recommended immunization schedules",
        "https://www.canada.ca/en/public-health/services/publications/healthy-living/"
        "canadian-immunization-guide-part-1-key-immunization-information/"
        "page-13-recommended-immunization-schedules.html",
    ),
    SourceDoc(
        "cig-populations",
        "Vaccination of specific populations",
        "https://www.canada.ca/en/public-health/services/publications/healthy-living/"
        "canadian-immunization-guide-part-3-vaccination-specific-populations.html",
    ),
    SourceDoc(
        "cig-adults",
        "Immunization of adults",
        "https://www.canada.ca/en/public-health/services/publications/healthy-living/"
        "canadian-immunization-guide-part-3-vaccination-specific-populations/page-2-immunization-of-adults.html",
    ),
    SourceDoc(
        "cig-influenza",
        "Influenza vaccines",
        "https://www.canada.ca/en/public-health/services/publications/healthy-living/"
        "canadian-immunization-guide-part-4-active-vaccines/page-10-influenza-vaccine.html",
    ),
    SourceDoc(
        "cig-hepb",
        "Hepatitis B vaccines",
        "https://www.canada.ca/en/public-health/services/publications/healthy-living/"
        "canadian-immunization-guide-part-4-active-vaccines/page-7-hepatitis-b-vaccine.html",
    ),
]
