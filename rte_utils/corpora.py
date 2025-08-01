import re
import pandas as pd

def umls_dicts(corpora_path):
    # Download 2024AB in CSV form from https://bioportal.bioontology.org/ontologies/STY?p=summary
    UMLS_TYPES = pd.read_csv(
        f"{corpora_path}/STY.csv"
    )
    UMLS_TYPES["id"] = UMLS_TYPES["Class ID"].str.split("/").str[-1]
    UMLS_TYPES["label"] = UMLS_TYPES["Preferred Label"].str[2:-2]

    UMLS_DICT = UMLS_TYPES.set_index("id")["label"].to_dict()
    UMLS_DICT = {
        key: "".join(
            [strseg.capitalize() for strseg in value.replace("-", " ").split(" ")]
        ).replace(",", "")
        for key, value in UMLS_DICT.items()
    }
    UMLS_DICT["UnknownType"] = ""


    # UMLS_MRSTY = pd.read_csv(
    #     f"{CORPORA_DATA}/MRSTY.RRF",
    #     sep="|",
    #     names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "empty"],
    #     index_col="CUI",  # is index
    # )


    # https://www.ncbi.nlm.nih.gov/books/NBK9685/#ch03.sec3.3.7
    UMLS_MRSTY = pd.read_csv(
        "https://www.nlm.nih.gov/research/umls/new_users/online_learning/data-files/MRSTY.RRF",
        sep="|",
        names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF"],
        index_col=None,  # is index
    )

    UMLS_CUI_DICT = UMLS_MRSTY["STN"].to_dict()
    UMLS_CUI_DICT = {
        key: "".join(
            [strseg.capitalize() for strseg in value.replace("-", " ").split(" ")]
        ).replace(",", "")
        for key, value in UMLS_CUI_DICT.items()
    }
    UMLS_CUI_DICT[""] = ""
    
    return UMLS_DICT, UMLS_CUI_DICT


def load_bigbio_datasets():
    from bigbio.dataloader import (
        BigBioConfigHelpers,
    )  # pylint: disable=import-error, wrong-import-position, no-name-in-module # type: ignore
    conhelps = BigBioConfigHelpers()
    
    bb_kb_datasets = {}
    bb_kb_helpers = conhelps.filtered(
        lambda x: x.is_bigbio_schema
        and x.config.name.endswith("_kb")
        and not x.is_local
        and not x.is_large
        and "Non Commercial" not in x.license
        and "cantemist" not in x.config.name  # Github LFS?
        and "ask_a_patient_bigbio_kb" != x.config.name  # not relevant
        and "medmentions_st21pv_bigbio_kb" != x.config.name  # Other subset used
        and "medmentions"
        not in x.config.name  # TODO: SELECT SUBSET OF ENTITY TYPES TO USE OR DONT USE
        and "meddocan_bigbio_kb" != x.config.name  # Not relevant
        and "muchmore"
        not in x.config.name  # Duplicate entity type definition is unavoidable
        and "spl_adr_200db_unannotated_bigbio_kb"
        != x.config.name  # Other subset used
        and "swedish_medical_ner_lt_bigbio_kb"
        != x.config.name  # Other subset used
        and "swedish_medical_ner_wiki_bigbio_kb"
        != x.config.name  # Other subset used
        and "linnaeus_bigbio_kb" != x.config.name  # Other subset used
        and "genetagcorrect_bigbio_kb" != x.config.name  # Other subset used
        and "tmvar_v3_bigbio_kb" != x.config.name  # Data load error
        and not (
            "mantra_gsc" in x.config.name and "_en_" not in x.config.name
        )  # English subset used
        and "jnlpba_bigbio_kb" != x.config.name
        and "minimayosrs_bigbio_pairs" not in x.config.name  # Too small
        # and "symptemist" not in x.config.name # Spanish
        and not (
            "symptemist" in x.config.name
            and "symptemist_linking_complete_bigbio_kb" != x.config.name
        )
        and "progene_bigbio_kb" != x.config.name  # Other subset used
        and "chebi_nactem_abstr_ann2"
        != x.config.name  # using chebi_nactem_abstr_ann1
        and "spl_adr_200db_train_bigbio_kb"
        != x.config.name
        and "lll_bigbio_kb" != x.config.name  # No annotations on test set
        and "bionlp_st_2013_gro_bigbio_kb" != x.config.name
        and "distemist_entities_bigbio_kb"
        != x.config.name  # Using distemist_linking_bigbio_kb
        and "chebi_nactem_abstr_ann2_bigbio_kb"
        != x.config.name  # Using chebi_nactem_abstr_ann1_bigbio_kb
        and "twadrl_bigbio_kb"
        not in x.config.name  # should we use? NO - social media types? split to subsets?
        and "bioid_bigbio_kb"
        != x.config.name  # TODO: Report for fix. Doesn't work now - only one entity per text.
        and not x.config.name.startswith(
            "bioscope_"
        )
    )
    
    # Load/download all kb datasets
    for helper in bb_kb_helpers:
        print(helper.config.name, helper.tasks)
        if helper.config.name not in bb_kb_datasets:
            if (
                helper.config.name in "biorelex_bigbio_kb"
                or helper.config.name.startswith("euadr")
                or helper.config.name.startswith("progene")
                or helper.config.name.startswith("complextome")
                or helper.config.name.startswith("regulatome")
                or helper.config.name.startswith("jnlpba")
            ):
                bb_kb_datasets[helper.config.name] = helper.load_dataset(
                    from_hub=False, trust_remote_code=True
                )
                continue
            bb_kb_datasets[helper.config.name] = helper.load_dataset(
                trust_remote_code=True
            )
    
    ### codiesp_X fix
    codiesp_X_src = conhelps.for_config_name("codiesp_X_source").load_dataset(
        trust_remote_code=True
    )

    for split in ("train", "validation", "test"):
        codiesp_X_src_dict = {
            row["document_id"]: row["text"] for row in codiesp_X_src[split]
        }

        def insert_codiesp_passage(example):
            example["passages"] = [
                {
                    "id": example["document_id"] + "_text",
                    "type": "case_report",
                    "text": [codiesp_X_src_dict[example["document_id"]]],
                    "offsets": [],
                }
            ]
            return example

        bb_kb_datasets["codiesp_X_bigbio_kb"][split] = bb_kb_datasets[
            "codiesp_X_bigbio_kb"
        ][split].map(insert_codiesp_passage)

    fixed_datasets = []

    for dname in bb_kb_datasets.keys():
        if (
            not (
                "test" in bb_kb_datasets[dname]
                or "validation" in bb_kb_datasets[dname]
            )
            and "train" in bb_kb_datasets[dname]
        ):
            fixed_datasets.append(dname)
            bb_kb_datasets[dname]["test"] = bb_kb_datasets[dname][
                "train"
            ].filter(lambda example, idx: idx % 2 == 0, with_indices=True)
            bb_kb_datasets[dname]["train"] = bb_kb_datasets[dname][
                "train"
            ].filter(lambda example, idx: idx % 2 != 0, with_indices=True)
            print(dname, "was split in half into a test set.")


    print("Fixed datasets:", fixed_datasets)
            
    return bb_kb_datasets, bb_kb_helpers
    
def create_bigbio_bibliography(bb_kb_helpers):
    ### CREATE BIBLIOGRAPHY FILE FROM CORPORA
    prev_name = None
    with open("bigbio.bib", "w", encoding="utf-8") as f:
        for helper in bb_kb_helpers:
            if helper.dataset_name != prev_name:
                print("%", helper.display_name, file=f)

                citation = "".join(helper.citation).strip("\n")
                citation = citation.replace("%", "\\%")
                start_paren = citation.count("{")
                end_paren = citation.count("}")
                if start_paren > end_paren:
                    citation += (start_paren - end_paren) * "}"

                citation_arr = citation.split("{", 1)
                citation_arr[1] = citation_arr[1].split(",\n", 1)
                citation_arr[1][0] = helper.dataset_name
                citation = (
                    citation_arr[0]
                    + "{bigbio:"
                    + citation_arr[1][0]
                    + ",\n"
                    + citation_arr[1][1]
                )

                print(citation, file=f)
                print("", file=f)
                prev_name = helper.dataset_name
                    
                    
 # FETCH VALID TYPES FOR EACH CONFIG
def fetch_type_set(bb_kb_datasets, config_name, element_type):
    from datasets import concatenate_datasets
    
    type_set = set()
    full_dataset = bb_kb_datasets[config_name]

    dataset = concatenate_datasets(
        [full_dataset[key] for key in full_dataset.keys()]
    )
    # dataset = None
    # if "train" in full_dataset:
    #     dataset = full_dataset['train']
    # if dataset is None and "validation" in full_dataset:
    #     dataset = full_dataset['validation']
    # if dataset is None and "test" in full_dataset:
    #     dataset = full_dataset['test']

    # if element_type not in dataset[0]

    for article in dataset:
        for rel in article[element_type]:
            type_set.add(rel["type"])
    return type_set

def standardise_type_name(type_name, element_type):
    if type_name == "":
        return ""
    elif len(type_name) == 1:
        return type_name.upper()

    type_name = type_name.replace(" ", "")
    type_name = type_name.replace(",", "")

    if element_type == "entities":
        if type_name.upper() == "SNP":
            return "SNP"
        if type_name.upper() == "ENTITY":
            return "ProteinRelatedEntity"
        type_name = type_name.replace("/", "Or")
        type_name = type_name.replace("-Y", "")
        type_name = type_name.replace("-N", "")
        type_name_split = re.split(r"_|-", type_name)
        formatted_name = "".join(
            [
                (
                    s.capitalize()
                    if s[0].islower() or (len(s) > 1 and s[1].isupper())
                    else s
                )
                for s in type_name_split
            ]
        )
        return formatted_name.replace("Rna", "RNA").replace("Dna", "DNA")
    else:
        if type_name.upper() == "PPI":
            return "PROTEIN_PROTEIN_INTERACTION"
        type_name = type_name.replace("/", "-OR-")
        type_name_split = re.split(r"_|-", type_name)
        return "_".join([s.upper() for s in type_name_split])

def fetch_type_names_dict(config_name, element_type, bb_kb_datasets):
    valid_type = {"entities", "relations", "events"}
    if element_type not in valid_type:
        raise ValueError("results: status must be one of %r." % valid_type)

    # if config_name == "lll_bigbio_kb":
    #     if element_type == "entities":
    #         return {"agent": "Protein", "target": "Gene"}
    #     if element_type == "relations":
    #         return {"genic_interaction": "PROTEIN_GENE_INTERACTION"}
    #     return dict()
    if config_name == "bioinfer_bigbio_kb" and element_type == "entities":
        return {
            "Individual_protein": "Protein",
            "Gene/protein/RNA": "GeneOrProteinOrRNA",
            "Protein_complex": "ProteinComplex",
            "DNA_family_or_group": "DNAFamilyOrGroup",
            "Gene": "Gene",
            "Protein_family_or_group": "ProteinFamilyOrGroup",
        }
    elif config_name == "bioid_bigbio_kb" and element_type == "entities":
        return {
            "molecule": "Molecule",
            "rna": "RNA",
            "cell": "Cell",
            "assay": "Assay",
            "chemical": "Chemical",
            "species": "Species",
            "cellline": "CellLine",
            "subcellular": "Subcellular",
            "protein": "Protein",
            "anatomy": "Anatomy",
            "organism": "Organism",
            "tissue": "Tissue",
            "gene": "Gene",
        }
    elif config_name == "bc5cdr_bigbio_kb" and element_type == "relations":
        return {"CID": "INDUCES_DISEASE"}
    elif config_name == "biorelex_bigbio_kb" and element_type == "entities":
        return {
            "fusion-protein": "FusionProtein",
            "amino-acid": "AminoAcid",
            "protein-complex": "ProteinComplex",
            "drug": "Drug",
            "mutation": "Mutation",
            "cell": "Cell",
            "chemical": "Chemical",
            "protein-region": "ProteinRegion",
            "protein-DNA-complex": "ProteinDNAComplex",
            "protein": "Protein",
            "RNA-family": "RNAFamily",
            "experimental-construct": "ExperimentalConstruct",
            "assay": "Assay",
            "organelle": "Organelle",
            "brand": "Brand",
            "disease": "Disease",
            "DNA": "DNA",
            "protein-domain": "ProteinDomain",
            "experiment-tag": "ExperimentTag",
            "process": "Process",
            "gene": "Gene",
            "protein-family": "ProteinFamily",
            "peptide": "Peptide",
            "protein-isoform": "ProteinIsoform",
            "tissue": "Tissue",
            "protein-motif": "ProteinMotif",
            "parameter": "Parameter",
            "protein-RNA-complex": "ProteinRNAComplex",
            "organism": "Organism",
            "RNA": "RNA",
            "reagent": "Reagent",
        }
    elif (
        config_name == "chem_dis_gene_bigbio_kb"
        and element_type == "relations"
    ):
        return {
            "chem_gene:increases^expression": "INCREASES_EXPRESSION_OF",
            "chem_gene:affects^binding": "AFFECTS_BINDING_OF",
            "gene_disease:therapeutic": "THERAPEUTIC_FOR",
            "chem_gene:affects^transport": "AFFECTS_TRANSPORT_OF",
            "chem_gene:decreases^metabolic_processing": "DECREASES_METABOLIC_PROCESSING_FOR",
            "chem_gene:increases^activity": "INCREASES_ACTIVITY_OF",
            "chem_gene:decreases^activity": "DECREASES_ACTIVITY_OF",
            "chem_gene:decreases^transport": "DECREASES_TRANSPORT_OF",
            "chem_gene:affects^expression": "AFFECTS_EXPRESSION_OF",
            "chem_gene:increases^metabolic_processing": "INCREASES_METABOLIC_PROCESSING_OF",
            "chem_gene:decreases^expression": "DECREASES_EXPRESSION_OF",
            "chem_gene:affects^metabolic_processing": "AFFECTS_METABOLIC_PROCESSING_OF",
            "chem_gene:affects^localization": "AFFECTS_LOCALIZATION_OF",
            "gene_disease:marker/mechanism": "MARKER_OR_MECHANISM_FOR",
            "chem_disease:marker/mechanism": "MARKER_OR_MECHANISM_FOR",
            "chem_disease:therapeutic": "THERAPEUTIC_FOR",
            "chem_gene:increases^transport": "INCREASES_TRANSPORT_OF",
            "chem_gene:affects^activity": "AFFECTS_ACTIVITY_OF",
        }
    elif config_name == "verspoor_2013_bigbio_kb":
        if element_type == "entities":
            return {
                "mutation": "Mutation",
                "disease": "Disease",
                "gender": "Gender",
                "age": "Age",
                "Concepts_Ideas": "Characteristic",
                "body-part": "BodyPart",
                "ethnicity": "Ethnicity",
                "Disorder": "Disorder",
                "cohort-patient": "CohortOrPatient",
                "size": "Size",
                "gene": "Gene",
                "Physiology": "Physiology",
                "Phenomena": "Characteristic",
            }
        if element_type == "relations":
            return {"relatedTo": "RELATED_TO", "has": "HAS"}
        return {}
    elif config_name.startswith("mantra_gsc") and element_type == "entities":
        return {
            "PROC": "Procedure",
            "LIVB": "LivingBeing",
            "DISO": "Disorder",
            "Manufactured Object": "",
            "Research Activity": "",
            "OBJC": "Object",
            "ANAT": "Anatomy",
            "PHYS": "Physiology",
            "Mental or Behavioral Dysfunction": "",
            "Research Device": "Device",
            "CHEM": "ChemicalOrDrug",
            "GEOG": "GeographicArea",
            "DEVI": "Device",
            "PHEN": "Phenomena",
            "Amino Acid, Peptide, or Protein|Enzyme|Receptor": "",
            "OBJC|CHEM": "ChemicalOrDrug",
        }
    elif config_name.startswith("quaero") and element_type == "entities":
        return {
            "PROC": "Procedure",
            "LIVB": "LivingBeing",
            "DISO": "Disorder",
            "OBJC": "Object",
            "ANAT": "Anatomy",
            "PHYS": "Physiology",
            "CHEM": "ChemicalOrDrug",
            "GEOG": "GeographicArea",
            "DEVI": "Device",
            "PHEN": "Phenomena",
        }
    elif config_name == "iepa_bigbio_kb" and element_type == "entities":
        return {"": "Protein"}
    elif (
        config_name == "ncbi_disease_bigbio_kb" and element_type == "entities"
    ):
        return {
            "DiseaseClass": "DiseaseClass",
            "CompositeMention": "CompositeDiseaseMention",
            "SpecificDisease": "SpecificDisease",
            "Modifier": "DiseaseModifier",
        }
    elif config_name == "nlm_gene_bigbio_kb" and element_type == "entities":
        return {
            "STARGENE": "Gene",
            "Domain": "ProteinDomain",
            "GENERIF": "Gene",
            "Gene": "Gene",
            "Other": "GeneProductOrMarkerGene",
        }  # https://www.sciencedirect.com/science/article/pii/S1532046421001088
    elif config_name == "osiris_bigbio_kb" and element_type == "entities":
        return {"variant": "GeneticVariant", "gene": "Gene"}
    elif config_name == "an_em_bigbio_kb" and element_type == "relations":
        return {
            # "frag": "RELATED_FRAGMENT_OF",
            "frag": "",
            "Part-of": "",
        }  # TODO: Consider removing RELATED_FRAGMENT_OF?
    elif (
        config_name == "scai_chemical_bigbio_kb" and element_type == "entities"
    ):
        return {
            "": "",
            "ABBREVIATION": "ChemicalAbbreviation",
            "FAMILY": "ChemicalFamily",
            "TRIVIAL": "Chemical",
            "PARTIUPAC": "PartChemical",
            "IUPAC": "Chemical",
            "TRIVIALVAR": "Chemical",
            "SUM": "Chemical",
            "MODIFIER": "",
        }  # TODO: BEWARE of entity names that are split up. 'text' field is list and contains multiple strs.
    elif (
        config_name == "scai_disease_bigbio_kb" and element_type == "entities"
    ):
        return {"DISEASE": "Disease", "ADVERSE": "AdverseEffect"}
    elif config_name == "seth_corpus_bigbio_kb":
        if element_type == "entities":
            return {"Gene": "Gene", "SNP": "SNP", "RS": "SNP"}
        if element_type == "relations":
            return {"AssociatedTo": "ASSOCIATED_TO", "Equals": "IS_EQUAL_TO"}
        return dict()
    elif config_name.startswith("tmvar") and element_type == "entities":
        return {
            "SNP": "SNP",
            "DNAMutation": "DNAMutation",
            "ProteinMutation": "ProteinMutation",
        }
    # elif config_name == "symptemist_linking_complete_bigbio_kb" and element_type == "entities":
    #     return {
    #         "Sintoma": "Symptom"
    #     }
    elif config_name == "pharmaconer_bigbio_kb" and element_type == "entities":
        return {
            "NO_NORMALIZABLES": "Quimico",
            "NORMALIZABLES": "Quimico",
            "PROTEINAS": "Proteina",
            "UNCLEAR": "",
        }
    elif config_name == "chia_bigbio_kb":
        if element_type == "entities":
            return {
                "Condition": "Condition",
                "Device": "",
                "Drug": "Drug",
                "Measurement": "",
                "Mood": "",
                "Multiplier": "",
                "Negation": "",
                "Observation": "",
                "Person": "",
                "Procedure": "Procedure",
                "Qualifier": "",
                "ReferencePoint": "",
                "Scope": "",
                "Temporal": "",
                "Value": "",
                "Visit": "",
            }
        return dict()
    elif config_name == "bionlp_st_2011_rel_bigbio_kb":
        if element_type == "entities":
            return {"Entity": "ProteinRelatedEntity", "Protein": "Protein"}
        if element_type == "relations":
            return {
                "Subunit-Complex": "",  # "FORMS_SUBUNIT_COMPLEX_WITH",
                "Protein-Component": "",  # "IS_PROTEIN_COMPONENT_OF",
            }
        return dict()
    elif (
        config_name == "bionlp_st_2013_ge_bigbio_kb"
        and element_type == "relations"
    ):
        return {"Coreference": ""}
    # elif config_name == "bionlp_st_2011_epi_bigbio_kb":
    #     if element_type == "entities":
    #         return {"Entity": "ProteinRelatedEntity", "Protein": "Protein"}
    #     if element_type == "relations":
    #         return {
    #             "Subunit-Complex": "FORMS_SUBUNIT_COMPLEX_WITH",
    #             "Protein-Component": "IS_PROTEIN_COMPONENT_OF",
    #         }
    #     return dict()
    elif (
        config_name == "citation_gia_test_collection_bigbio_kb"
        or config_name == "gnormplus_bigbio_kb"
    ) and element_type == "entities":
        return {
            "Gene": "GeneOrProtein",
            "FamilyName": "GeneOrProteinFamily",
            "DomainMotif": "ProteinDomain",
        }
    elif config_name == "genetaggold_bigbio_kb" and element_type == "entities":
        return {"NEWGENE": "Gene"}
    elif config_name == "coneco_bigbio_kb" and element_type == "entities":
        return {"Complex": "Complex"}
    elif (
        config_name == "bionlp_shared_task_2009_bigbio_kb"
        and element_type == "entities"
    ):
        return {"Entity": "ProteinRelatedEntity", "Protein": "Protein"}
    elif config_name == "biored_bigbio_kb" and element_type == "relations":
        return {
            "Association": "ASSOCIATED_WITH",
            "Conversion": "CONVERTED_TO",
            "Comparison": "COMPARED_TO",
            "Cotreatment": "IN_COTREATMENT_WITH",
            "Positive_Correlation": "POSITIVELY_CORRELATED_WITH",
            "Negative_Correlation": "NEGATIVELY_CORRELATED_WITH",
            "Drug_Interaction": "HAS_DRUG_INTERACTION_WITH",
            "Bind": "BINDS",
        }
    elif config_name == "euadr_bigbio_kb":
        if element_type == "entities":
            return {
                "Chemicals & Drugs": "ChemicalOrDrug",
                "Diseases & Disorders": "DiseaseOrDisorder",
                "Genes & Molecular Sequences": "GeneOrMolecularSequence",
                "SNP & Sequence variations": "SNPOrSequenceVariation",
                "": "",
            }
        if element_type == "relations":
            return {
                "PA": "POSITIVE_ASSOCIATION",
                "NA": "NEGATIVE_ASSOCIATION",
                "SA": "SPECULATIVE_ASSOCIATION",
            }
        return dict()
    elif config_name == "complextome_bigbio_kb":
        if element_type == "entities":
            return {
                "Family": "ProteinFamily",
                "Complex": "Complex",
                "Protein": "GeneOrProtein",
                "Chemical": "Chemical",
            }
        if element_type == "relations":
            return {"Complex_formation": "IN_COMPLEX_WITH"}
        return dict()
    elif config_name == "regulatome_bigbio_kb":
        if element_type == "entities":
            return {
                "Family": "ProteinFamily",
                "Complex": "Complex",
                "Protein": "GeneOrProtein",
                "Chemical": "Chemical",
            }
        if element_type == "relations":
            return {
                "Catalysis_of_deneddylation": "CATALYSES_DENEDDYLATION_OF",  #
                "Regulation_of_gene_expression": "REGULATES_GENE_EXPRESSION_OF",  #
                "Regulation": "",  #
                "Catalysis_of_deglycosylation": "CATALYSES_DEGLYCOSYLATION_OF",  #
                "Catalysis_of_methylation": "CATALYSES_METHYLATION_OF",  #
                "Other_catalysis_of_small_molecule_conjugation": "CATALYSES_OTHER_SMALL_MOLECULE_CONJUGATION_TO",  #
                "Other_catalysis_of_small_molecule_removal": "CATALYSES_OTHER_SMALL_MOLECULE_REMOVAL_FROM",  #
                "Other_catalysis_of_small_protein_conjugation": "CATALYSES_OTHER_SMALL_PROTEIN_CONJUGATION_TO",  #
                "Catalysis_of_posttranslational_modification": "",
                "Catalysis_of_acylation": "CATALYSES_ACYLATION_OF",  #
                "Catalysis_of_neddylation": "CATALYSES_NEDDYLATION_OF",  #
                "Catalysis_of_small_protein_conjugation_or_removal": "",
                "Catalysis_of_demethylation": "CATALYSES_DEMETHYLATION_OF",  #
                "Catalysis_of_phosphoryl_group_conjugation_or_removal": "",
                "Catalysis_of_deacetylation": "CATALYSES_DEACETYLATION_OF",
                "Catalysis_of_ubiquitination": "CATALYSES_UBIQUITINATION_OF",
                "Complex_formation": "FORMS_COMPLEX_WITH",
                "Catalysis_of_palmitoylation": "CATALYSES_PALMITOYLATION_OF",
                "Catalysis_of_small_protein_conjugation": "",
                "Regulation_of_translation": "REGULATES_TRANSLATION_OF",
                "Catalysis_of_glycosylation": "CATALYSES_GLYCOSYLATION_OF",
                "Catalysis_of_deacylation": "CATALYSES_DEACYLATION_OF",
                "Catalysis_of_phosphorylation": "CATALYSES_PHOSPHORYLATION_OF",
                "Catalysis_of_prenylation": "CATALYSES_PRENYLATION_OF",
                "Other_catalysis_of_small_protein_removal": "CATALYSES_OTHER_SMALL_PROTEIN_REMOVAL_FROM",
                "Catalysis_of_acetylation": "CATALYSES_ACETYLATION_OF",
                "Catalysis_of_deSUMOylation": "CATALYSES_DESUMOYLATION_OF",
                "Catalysis_of_other_small_molecule_conjugation_or_removal": "",
                "Positive_regulation": "POSITIVELY_REGULATES",
                "Catalysis_of_farnesylation": "CATALYSES_FARNESYLATION_OF",
                "Catalysis_of_deubiquitination": "CATALYSES_DEUBIQUITINATION_OF",
                "Catalysis_of_geranylgeranylation": "CATALYSES_GERANYLGERANYLATION_OF",
                "Regulation_of_transcription": "REGULATES_TRANSCRIPTION_OF",
                "Catalysis_of_depalmitoylation": "CATALYSES_DEPALMITOYLATION_OF",
                "Regulation_of_degradation": "REGULATES_DEGRADATION_OF",
                "Negative_regulation": "NEGATIVELY_REGULATES",
                "Catalysis_of_lipidation": "CATALYSES_LIPIDATION_OF",
                "Catalysis_of_dephosphorylation": "CATALYSES_DEPHOSPHORYLATION_OF",
                "Catalysis_of_small_protein_removal": "",
                "Catalysis_of_small_molecule_removal": "",
                "Catalysis_of_ADP-ribosylation": "CATALYSES_ADP_RIBOSYLATION_OF",
                "Catalysis_of_SUMOylation": "CATALYSES_SUMOYLATION_OF",
            }
        return dict()
    elif (
        config_name == "genia_term_corpus_bigbio_kb"
        and element_type == "entities"
    ):
        type_set = fetch_type_set(bb_kb_datasets, config_name, element_type)
        return {
            name: standardise_type_name(name, element_type)
            for name in type_set
            if ("(" not in name and "other" not in name and "N/A" not in name)
        }
    elif config_name == "chemprot_bigbio_kb":
        if element_type == "entities":
            return {
                "CHEMICAL": "Chemical",
                "GENE-Y": "GeneOrProtein",
                "GENE-N": "GeneOrProtein",
            }
        if element_type == "relations":
            return {
                "Not": "",
                "Agonist": "IS_AGONIST_FOR",
                "Upregulator": "UPREGULATES",
                "Part_of": "",
                "Modulator": "",
                "Regulator": "",
                "Cofactor": "",
                "Antagonist": "IS_ANTAGONIST_FOR",
                "Substrate": "IS_SUBSTRATE_FOR",
                "Downregulator": "DOWNREGULATES",
                "Undefined": "",
            }
        return dict()
    elif (
        config_name == "medmentions_full_bigbio_kb"
        and element_type == "entities"
    ):
        type_set = fetch_type_set(bb_kb_datasets, config_name, element_type)
        return {name: UMLS_DICT.get(name, "") for name in type_set}
    elif config_name in non_re_re_sets and element_type == "relations":
        type_set = fetch_type_set(bb_kb_datasets, config_name, element_type)
        return dict.fromkeys(type_set, "")
    else:
        type_set = fetch_type_set(bb_kb_datasets, config_name, element_type)
        return {
            name: standardise_type_name(name, element_type)
            for name in type_set
        }
        
def fix_rel_type_verspoor(head_type, tail_type, rev=False):
        if tail_type == "Mutation":
            return "HAS_MUTATION"
        if head_type == "Mutation" and tail_type == "Disease":
            return "VARIANT_ASSOCIATED_WITH"
        if head_type == "Mutation" and tail_type == "Size":
            return "HAS_MUTATION_FREQUENCY"
        if head_type == "Disease" and tail_type in (
            "Characteristic",
            "Physiology",
            "Disorder",
        ):
            return "HAS_DISEASE_CHARACTERISTIC"
        if tail_type == "Gene" and head_type in (
            "Characteristic",
            "Physiology",
            "Disorder",
        ):
            return "RELATED_TO"
        if head_type == "Disease" and tail_type == "Gene":
            return "HAS_GENETIC_ASSOCIATION"
        if head_type == "Disease" and tail_type == "BodyPart":
            return "DISEASE_OCCURS_IN"
        if head_type == "CohortOrPatient" and tail_type == "Age":
            return "HAS_AGE"
        if head_type == "CohortOrPatient" and tail_type == "Gender":
            return "HAS_GENDER"
        if head_type == "CohortOrPatient" and tail_type == "Ethnicity":
            return "HAS_ETHNICITY"
        if head_type == "CohortOrPatient" and tail_type == "Disease":
            return "HAS_DISEASE"
        if head_type == "CohortOrPatient" and tail_type in (
            "Characteristic",
            "Physiology",
            "Disorder",
        ):
            return "HAS_COHORT_CHARACTERISTIC"
        if head_type == "CohortOrPatient" and tail_type == "Size":
            return "HAS_COHORT_SIZE"

        if rev:
            return ""

        return (
            fix_rel_type_verspoor(tail_type, head_type, rev=True),
            True,
        )  # Reversed


    def parse_bigbio_to_llm(helper, split_name):
        config_name = helper.config.name

        if split_name in bb_kb_datasets[config_name]:
            dataset = bb_kb_datasets[config_name][
                split_name
            ]  # TODO: ADD Validation and fixed train_x
        else:
            print(f"No {split_name} split found for {config_name}")
            return

        (
            ENTITY_TRANSLATION_DICT,
            RELATION_TRANSLATION_DICT,
            EVENT_TRANSLATION_DICT,
        ) = translation_dicts[config_name]

        parsed_dataset = dict()
        text_types = list()

        for article in dataset:
            article_id = article["id"]
            article_text_arr = []
            passage_types = []
            for passage in article["passages"]:
                passage_types.append(passage["type"])
                article_text_arr.extend(
                    passage["text"]
                )  ##TODO: Remove "** IGNORE LINE ** from verspoor"

            passage_types_str = ",".join(passage_types)
            text_types.append(passage_types_str)

            article_text = "".join(article_text_arr).strip("\n")

            make_relation_dicts = bool(article["relations"])

            args_dict = dict()
            normalized_to_first_id_dict = dict()

            entities = dict()
            for entity in article["entities"]:
                if config_name in (
                    "muchmore_en_bigbio_kb",
                    "muchmore_de_bigbio_kb",
                ):
                    entity_type = UMLS_CUI_DICT.get(
                        entity["normalized"][0]["db_id"], ""
                    )
                    # print(entity, "result:", entity_type)
                else:
                    entity_type = ENTITY_TRANSLATION_DICT.get(entity["type"], "")
                if entity_type == "":
                    if config_name == "euadr_bigbio_kb":
                        entity_name = "".join(entity["text"])
                        inferred_entity = None
                        for entity_compare in entities:
                            if entity_compare[0] == entity_name:
                                inferred_entity = entity_compare
                                break
                        if inferred_entity is not None:
                            args_dict[entity["id"]] = inferred_entity
                        continue
                    continue

                entity_name = "".join(entity["text"])

                found_entity_id = None
                if entity["normalized"] and config_name not in (
                    "muchmore_en_bigbio_kb",
                    "muchmore_de_bigbio_kb",
                ):
                    for normalized in entity[
                        "normalized"
                    ]:  ###TODO: DO THE SAME for datasets W/O NED
                        normalized_tuple = (
                            normalized["db_name"],
                            normalized["db_id"],
                        )
                        if normalized_tuple not in normalized_to_first_id_dict:
                            normalized_to_first_id_dict[normalized_tuple] = entity[
                                "id"
                            ]
                            entities[(entity_name, entity_type)] = (
                                None  # TODO: ADD NED feature
                            )
                            # TODO: Make it add anyway if it is significantly different vector dist.
                            # (in the case of abbreviations etc.)
                            # cf. also here coreferences field in e.g. bionlp_shared_tasks_2009
                        elif (
                            normalized_to_first_id_dict[normalized_tuple]
                            != entity["id"]
                        ):
                            found_entity_id = normalized_to_first_id_dict[
                                normalized_tuple
                            ]
                            break
                else:
                    entities[(entity_name, entity_type)] = None

                if make_relation_dicts:
                    if found_entity_id is not None:
                        args_dict[entity["id"]] = args_dict[found_entity_id]
                    else:
                        args_dict[entity["id"]] = (entity_name, entity_type)

            entities = list(
                entities.keys()
            )  # TODO: If information wanted to be included later: just omit.

            relations = []
            previous_relations = set()
            for relation in article["relations"]:
                head_id = relation["arg1_id"]
                tail_id = relation[
                    "arg2_id"
                ]  # TODO: CHECK FOR MORE ARGS??? ESPECIALLY IN EVENT!!!
                if head_id in args_dict and tail_id in args_dict:
                    rel_type = RELATION_TRANSLATION_DICT.get(relation["type"], "")

                    if rel_type == "":
                        continue
                    if (
                        config_name == "verspoor_2013_bigbio_kb"
                        and rel_type != "RELATED_TO"
                    ):
                        rel_type = fix_rel_type_verspoor(
                            args_dict[head_id][1], args_dict[tail_id][1]
                        )  # TODO: FIX EVEN MORE
                        if len(rel_type) == 2:
                            rel_type, _ = rel_type
                            if rel_type != "":
                                rel_triple = (
                                    args_dict[tail_id],
                                    rel_type,
                                    args_dict[head_id],
                                )
                        elif rel_type != "":
                            rel_triple = (
                                args_dict[head_id],
                                rel_type,
                                args_dict[tail_id],
                            )
                    else:
                        rel_triple = (
                            args_dict[head_id],
                            rel_type,
                            args_dict[tail_id],
                        )

                    if rel_triple not in previous_relations:
                        previous_relations.add(rel_triple)
                        relations.append(rel_triple)

                # TODO: ADD RE feature

            events = []
            for event in article[
                "events"
            ]:  ##TODO: INVESTIGATE DIRECTIONALITY FOR EVENT + ARGUMENT STRUCTURE
                pass  # TODO: ADD RE feature

            parsed_dataset[article_id] = (
                article_text,
                entities,
                relations,
                events,
            )

        return parsed_dataset, text_types

