# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""A file containing prompt definitions."""
from __future__ import annotations


BIOGRAPH_EXTRACTION_PROMPT_NO_ICL = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relationship_type: One of the following types: [{relationship_types}]
- relationship_description: Explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Relationship_types: EXECUTIVE_OF,BOARD_MEMBER_OF
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}EXECUTIVE_OF{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}BOARD_MEMBER_OF{tuple_delimiter}Martin Smith is the Chair of the Central Institution and is thus a member of their board{tuple_delimiter}9)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Relationship_types: {relationship_types}
Text: {{query}}
######################
Output:"""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present{tuple_delimiter}5)
{completion_delimiter}

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Firuzabad held Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
{record_delimiter}
("entity"{tuple_delimiter}TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of Firuzabad where the Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in Quintara)
{record_delimiter}
("entity"{tuple_delimiter}CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}Aurelian who spent time in Tiruzia's Alhamia Prison)
{record_delimiter}
("entity"{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}FIRUZABAD{tuple_delimiter}AURELIA{tuple_delimiter}Firuzabad negotiated a hostage exchange with Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}AURELIA{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}Samuel Namara was a prisoner at Alhamia prison{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Samuel Namara and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Samuel Namara was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}FIRUZABAD{tuple_delimiter}Meggie Tazbah was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}FIRUZABAD{tuple_delimiter}Durke Bataglani was a hostage in Firuzabad{tuple_delimiter}2)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

BIOGRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relationship_type: One of the following types: [{relationship_types}]
- relationship_description: Explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Relationship_types: EXECUTIVE_OF,BOARD_MEMBER_OF
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}EXECUTIVE_OF{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}BOARD_MEMBER_OF{tuple_delimiter}Martin Smith is the Chair of the Central Institution and is thus a member of their board{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Relationship_types: OWNED,OWNS
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}OWNED{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present{tuple_delimiter}5)
{completion_delimiter}

######################
Example 3:
Entity_types: ProteinOrGene,DiseaseOrCondition,Phenotype,BiologicalProcess,TissueOrCell
Relationship_types: SYNONYM,INTERACTS_WITH,IS_TYPE_OF,ASSOCIATED_WITH,INCREASES,REDUCES
Text: Growth-arrest specific 6 (GAS6) is a secreted protein that acts as a ligand for TAM receptors (TYRO3, AXL and MERTK). In humans, GAS6 circulating levels and genetic variations in GAS6 are associated with hyperglycemia and increased risk of type 2 diabetes. However, the mechanisms by which GAS6 influences glucose metabolism are not understood. Here, we show that Gas6 deficiency in mice increases insulin sensitivity and protects from diet-induced insulin resistance. Conversely, increasing GAS6 circulating levels is sufficient to reduce insulin sensitivity in vivo. GAS6 inhibits the activation of the insulin receptor (IR) and reduces insulin response in muscle cells in vitro and in vivo. Mechanistically, AXL and IR form a complex, while GAS6 reprograms signaling pathways downstream of IR. This results in increased IR endocytosis following insulin treatment. This study contributes to a better understanding of the cellular and molecular mechanisms by which GAS6 and AXL influence insulin sensitivity.
######################
Output:
("entity"{tuple_delimiter}GAS6{tuple_delimiter}ProteinOrGene{tuple_delimiter}Growth-arrest specific 6 (GAS6) is a secreted protein)
{record_delimiter}
("entity"{tuple_delimiter}Growth-arrest specific 6{tuple_delimiter}ProteinOrGene{tuple_delimiter}Growth-arrest specific 6 (GAS6) is a secreted protein)
{record_delimiter}
("entity"{tuple_delimiter}TAM receptors{tuple_delimiter}ProteinOrGene{tuple_delimiter}TYRO3, AXL and MERTK are TAM receptors)
{record_delimiter}
("entity"{tuple_delimiter}TYRO3{tuple_delimiter}ProteinOrGene{tuple_delimiter}TYRO3, AXL and MERTK are TAM receptors)
{record_delimiter}
("entity"{tuple_delimiter}AXL{tuple_delimiter}ProteinOrGene{tuple_delimiter}TYRO3, AXL and MERTK are TAM receptors)
{record_delimiter}
("entity"{tuple_delimiter}MERTK{tuple_delimiter}ProteinOrGene{tuple_delimiter}TYRO3, AXL and MERTK are TAM receptors)
{record_delimiter}
("entity"{tuple_delimiter}hyperglycemia{tuple_delimiter}DiseaseOrCondition{tuple_delimiter}In humans, GAS6 circulating levels and genetic variations in GAS6 are associated with hyperglycemia)
{record_delimiter}
("entity"{tuple_delimiter}type 2 diabetes{tuple_delimiter}DiseaseOrCondition{tuple_delimiter}In humans, GAS6 circulating levels and genetic variations in GAS6 are associated with increased risk of type 2 diabetes)
{record_delimiter}
("entity"{tuple_delimiter}Gas6 deficiency{tuple_delimiter}DiseaseOrCondition{tuple_delimiter}Gas6 deficiency in mice increases insulin sensitivity and protects from diet-induced insulin resistance)
{record_delimiter}
("entity"{tuple_delimiter}insulin sensitivity{tuple_delimiter}Phenotype{tuple_delimiter}Gas6 deficiency in mice increases insulin sensitivity and protects from diet-induced insulin resistance)
{record_delimiter}
("entity"{tuple_delimiter}insulin resistance{tuple_delimiter}DiseaseOrCondition{tuple_delimiter}Gas6 deficiency in mice increases insulin sensitivity and protects from diet-induced insulin resistance)
{record_delimiter}
("entity"{tuple_delimiter}activation of the insulin receptor{tuple_delimiter}BiologicalProcess{tuple_delimiter}GAS6 inhibits the activation of the insulin receptor (IR))
{record_delimiter}
("entity"{tuple_delimiter}insulin receptor{tuple_delimiter}ProteinOrGene{tuple_delimiter}GAS6 inhibits the activation of the insulin receptor (IR))
{record_delimiter}
("entity"{tuple_delimiter}IR{tuple_delimiter}ProteinOrGene{tuple_delimiter}GAS6 inhibits the activation of the insulin receptor (IR))
{record_delimiter}
("entity"{tuple_delimiter}insulin response{tuple_delimiter}BiologicalProcess{tuple_delimiter}GAS6 reduces insulin response in muscle cells in vitro and in vivo)
{record_delimiter}
("entity"{tuple_delimiter}muscle{tuple_delimiter}TissueOrCell{tuple_delimiter}GAS6 reduces insulin response in muscle cells in vitro and in vivo)
{record_delimiter}
("entity"{tuple_delimiter}signaling pathways downstream of IR{tuple_delimiter}BiologicalProcess{tuple_delimiter}GAS6 reprograms signaling pathways downstream of IR)
{record_delimiter}
("entity"{tuple_delimiter}IR endocytosis{tuple_delimiter}BiologicalProcess{tuple_delimiter}There is increased IR endocytosis following insulin treatment)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}Growth-arrest specific 6{tuple_delimiter}SYNONYM{tuple_delimiter}Growth-arrest specific 6 (GAS6) is a secreted protein{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}TAM receptors{tuple_delimiter}INTERACTS_WITH{tuple_delimiter}Growth-arrest specific 6 (GAS6) is a secreted protein that acts as a ligand for TAM receptors{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}TYRO3{tuple_delimiter}TAM receptors{tuple_delimiter}IS_TYPE_OF{tuple_delimiter}TAM receptors (TYRO3, AXL and MERTK){tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}AXL{tuple_delimiter}TAM receptors{tuple_delimiter}IS_TYPE_OF{tuple_delimiter}TAM receptors (TYRO3, AXL and MERTK){tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}MERTK{tuple_delimiter}TAM receptors{tuple_delimiter}IS_TYPE_OF{tuple_delimiter}TAM receptors (TYRO3, AXL and MERTK){tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}hyperglycemia{tuple_delimiter}ASSOCIATED_WITH{tuple_delimiter}In humans, GAS6 circulating levels and genetic variations in GAS6 are associated with hyperglycemia{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}type 2 diabetes{tuple_delimiter}ASSOCIATED_WITH{tuple_delimiter}In humans, GAS6 circulating levels and genetic variations in GAS6 are associated with increased risk of type 2 diabetes{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6 deficiency{tuple_delimiter}insulin sensitivity{tuple_delimiter}INCREASES{tuple_delimiter}Gas6 deficiency in mice increases insulin sensitivity{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6 deficiency{tuple_delimiter}type 2 diabetes{tuple_delimiter}ASSOCIATED_WITH{tuple_delimiter}Gas6 deficiency in mice protects from diet-induced insulin resistance{tuple_delimiter}6)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}insulin sensitivity{tuple_delimiter}REDUCES{tuple_delimiter}Increasing GAS6 circulating levels is sufficient to reduce insulin sensitivity in vivo{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}activation of the insulin receptor{tuple_delimiter}REDUCES{tuple_delimiter}GAS6 inhibits the activation of the insulin receptor (IR){tuple_delimiter}6)
{record_delimiter}
("relationship"{tuple_delimiter}activation of the insulin receptor{tuple_delimiter}insulin receptor{tuple_delimiter}INCREASES{tuple_delimiter}Activation of the insulin receptor is a process that increases the activity insulin receptor{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}insulin response{tuple_delimiter}REDUCES{tuple_delimiter}GAS6 reduces insulin response in muscle cells in vitro and in vivo{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}insulin receptor{tuple_delimiter}IR{tuple_delimiter}SYNONYM{tuple_delimiter}insulin receptor (IR){tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}AXL{tuple_delimiter}IR{tuple_delimiter}INTERACTS_WITH{tuple_delimiter}Mechanistically, AXL and IR form a complex{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}signaling pathways downstream of IR{tuple_delimiter}ASSOCIATED_WITH{tuple_delimiter}GAS6 reprograms signaling pathways downstream of IR{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}GAS6{tuple_delimiter}IR endocytosis{tuple_delimiter}INCREASES{tuple_delimiter}Increased IR endocytosis following insulin treatment{tuple_delimiter}6)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Relationship_types: {relationship_types}
Text: {{query}}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed.  Answer YES | NO if there are still entities or relationships that need to be added.\n"


CORPUS_SEARCH_PROMPT = """As a biomedical researcher, you are able to extract the entities and relationships from a given abstract in the mentioned format
The allowed entity types are: ProteinOrGene, AlleleOrSNP, DiseaseOrCondition, SurgeryOrTherapyOrTreatment, MedicalProtocolOrMethodOrModel, BiologicalProcess, DrugOrChemical, Metabolite, Phenotype, Pathway.
Relationships should only be generated for results and conclusions and NOT for experiment setup, background or speculative or hypothesized relationships.
The allowed relationship types are: INTERACTS_WITH, ASSOCIATED_WITH, INVERSELY_ASSOCIATED_WITH, NOT_ASSOCIATED_WITH, RELATED_TO, NOT_RELATED_TO, UPREGULATES, UPREGULATED_IN, DOWNREGULATES, DOWNREGULATED_IN, REGULATES, REGULATED_IN, BINDS_TO, IN_COMPLEX_WITH, INCREASES, NOT_INCREASES, INCREASED_IN, DECREASES, NOT_DECREASES, DECREASED_IN, CAUSES, NOT_CAUSES, INHIBITS, ACTIVATES, PART_OF_PATHWAY, PROMOTES, NOT_PROMOTES, ACTIVATES, NOT_ACTIVATES, ADDS_POST_TRANSLATIONAL_MODIFICATION, REMOVES_POST_TRANSLATIONAL_MODIFICATION, LOCALIZED_TO, LOSS_OR_DEFICIENCY_OF, GAIN_OR_EXCESS_OF, PROTECTS_FROM, AFFECTS, NOT_AFFECTS, IMPROVES, NOT_IMPROVES, IMPROVED_IN, IMPAIRS, NOT_IMPAIRS, IMPAIRED_IN, IS_TYPE_OF, SYNONYM.
For the relationships, please provide the tissue, organism, or additional context in which the relationship was observed, if applicable.

The output should look like:
Entities:
Entity1 (EntityType)
Entity2 (EntityType)

Relationships:
Entity1 (EntityType) --RELATIONSHIP_TYPE-- Entity2 (EntityType) || tissue: Tissue; organism: Organism; context: Context


Here follows a few examples:

Example 1:
Text: Growth-arrest specific 6 (GAS6) is a secreted protein that acts as a ligand for TAM receptors (TYRO3, AXL and MERTK). In humans, GAS6 circulating levels and genetic variations in GAS6 are associated with hyperglycemia and increased risk of type 2 diabetes. However, the mechanisms by which GAS6 influences glucose metabolism are not understood. Here, we show that Gas6 deficiency in mice increases insulin sensitivity and protects from diet-induced insulin resistance. Conversely, increasing GAS6 circulating levels is sufficient to reduce insulin sensitivity in vivo. GAS6 inhibits the activation of the insulin receptor (IR) and reduces insulin response in muscle cells in vitro and in vivo. Mechanistically, AXL and IR form a complex, while GAS6 reprograms signaling pathways downstream of IR. This results in increased IR endocytosis following insulin treatment. This study contributes to a better understanding of the cellular and molecular mechanisms by which GAS6 and AXL influence insulin sensitivity.

Entities:
GAS6 (ProteinOrGene)
Growth-arrest specific 6 (ProteinOrGene)
TAM receptors (ProteinOrGene)
TYRO3 (ProteinOrGene)
AXL (ProteinOrGene)
MERTK (ProteinOrGene)
hyperglycemia (DiseaseOrCondition)
risk of type 2 diabetes (DiseaseOrCondition)
Gas6 deficiency (DiseaseOrCondition)
insulin resistance (DiseaseOrCondition)
insulin sensitivity (Phenotype)
activation of the insulin receptor (BiologicalProcess)
insulin response (BiologicalProcess)
muscle (TissueOrCell)
insulin receptor (ProteinOrGene)
IR (ProteinOrGene)
signaling pathways downstream of IR (BiologicalProcess)
IR endocytosis (BiologicalProcess)

Relationships:
GAS6 (ProteinOrGene) --SYNONYM-- Growth-arrest specific 6 (ProteinOrGene)
GAS6 (ProteinOrGene) --INTERACTS_WITH-- TAM receptors (ProteinOrGene)
TYRO3 (ProteinOrGene) --IS_TYPE_OF-- TAM receptors (ProteinOrGene)
AXL (ProteinOrGene) --IS_TYPE_OF-- TAM receptors (ProteinOrGene)
MERTK (ProteinOrGene) --IS_TYPE_OF-- TAM receptors (ProteinOrGene)
GAS6 (ProteinOrGene) --ASSOCIATED_WITH-- hyperglycemia (DiseaseOrCondition) || organism: humans
GAS6 (ProteinOrGene) --ASSOCIATED_WITH-- risk of type 2 diabetes (Phenotype) || organism: humans
Gas6 deficiency (DiseaseOrCondition) --LOSS_OR_DEFICIENCY_OF-- GAS6 (ProteinOrGene) || organism: mice
Gas6 deficiency (DiseaseOrCondition) --INCREASES-- insulin sensitivity (Phenotype) || organism: mice
Gas6 deficiency (DiseaseOrCondition) --PROTECTS_FROM-- insulin resistance (DiseaseOrCondition) || organism: mice
GAS6 (ProteinOrGene) --REDUCES-- insulin sensitivity (Phenotype) || organism: mice
GAS6 (ProteinOrGene) --INHIBITS-- activation of the insulin receptor (BiologicalProcess)
activation of the insulin receptor (BiologicalProcess) --ACTIVATES-- insulin receptor (ProteinOrGene)
GAS6 (ProteinOrGene) --REDUCES-- insulin response (BiologicalProcess) || tissue: muscle; organism: in vitro
GAS6 (ProteinOrGene) --REDUCES-- insulin response (BiologicalProcess) || tissue: muscle; organism: in vivo
insulin receptor (ProteinOrGene) --SYNONYM-- IR (ProteinOrGene)
AXL (ProteinOrGene) --IN_COMPLEX_WITH-- IR (ProteinOrGene)
GAS6 (ProteinOrGene) --AFFECTS-- signaling pathways downstream of IR (BiologicalProcess)
GAS6 (ProteinOrGene) --INCREASES-- IR endocytosis (BiologicalProcess) || context: following insulin treatment


Example 2:
Text: Maternal hypoxia is strongly linked to insulin resistance (IR) in adult offspring, and altered insulin signaling for muscle glucose uptake is thought to play a central role. However, whether the SIRT3/GSK-3β/GLUT4 axis is involved in maternal hypoxia-induced skeletal muscle IR in old male rat offspring has not been investigated. Maternal hypoxia was established from Days 5 to 21 of pregnancy by continuous infusion of nitrogen and air. The biochemical parameters and levels of key insulin signaling molecules of old male rat offspring were determined through a series of experiments. Compared to the control (Ctrl) old male rat offspring group, the hypoxic (HY) group exhibited elevated fasting blood glucose (FBG) (∼30%), fasting blood insulin (FBI) (∼35%), total triglycerides (TGs), and low-density lipoprotein cholesterol (LDL-C), as well as results showing impairment in the glucose tolerance test (GTT) and insulin tolerance test (ITT). In addition, hematoxylin-eosin (HE) staining and transmission electron microscopy (TEM) revealed impaired cellular structures and mitochondria in the longitudinal sections of skeletal muscle from HY group mice, which might be associated with decreased SIRT3 expression. Furthermore, the expression of insulin signaling molecules, such as GSK-3β and GLUT4, was also altered. In conclusion, the present results indicate that the SIRT3/GSK-3β/GLUT4 axis might be involved in maternal hypoxia-induced skeletal muscle IR in old male rat offspring.

Entities:
SIRT3/GSK-3β/GLUT4 axis (Pathway)
maternal hypoxia-induced skeletal muscle IR (DiseaseOrCondition)
Maternal hypoxia (DiseaseOrCondition)
insulin resistance (DiseaseOrCondition)
IR (DiseaseOrCondition)
fasting blood glucose (Phenotype)
fasting blood insulin (Phenotype)
total triglycerides (Phenotype)
low-density lipoprotein cholesterol (Phenotype)
glucose tolerance test (Phenotype)
insulin tolerance test (Phenotype)
SIRT3 (ProteinOrGene)
GSK-3β (ProteinOrGene)
GLUT4 (ProteinOrGene)

Relationships:
Maternal hypoxia (DiseaseOrCondition) --ASSOCIATED_WITH-- insulin resistance (DiseaseOrCondition) || organism: rats
insulin resistance (DiseaseOrCondition) --SYNONYM-- IR (DiseaseOrCondition)
Maternal hypoxia (DiseaseOrCondition) --INCREASES-- fasting blood glucose (Phenotype) || organism: rats; context: old male, hypoxic (HY) group vs. control
Maternal hypoxia (DiseaseOrCondition) --INCREASES-- fasting blood insulin (Phenotype) || organism: rats; context: old male, hypoxic (HY) group vs. control
Maternal hypoxia (DiseaseOrCondition) --INCREASES-- total triglycerides (Phenotype) || organism: rats; context: old male, hypoxic (HY) group vs. control
Maternal hypoxia (DiseaseOrCondition) --INCREASES-- low-density lipoprotein cholesterol (Phenotype) || organism: rats; context: old male, hypoxic (HY) group vs. control
Maternal hypoxia (DiseaseOrCondition) --IMPAIRS-- glucose tolerance test (Phenotype) || organism: rats; context: old male, hypoxic (HY) group vs. control
Maternal hypoxia (DiseaseOrCondition) --IMPAIRS-- insulin tolerance test (Phenotype) || organism: rats; context: old male, hypoxic (HY) group vs. control
SIRT3 --ASSOCIATED_WITH-- impaired cellular structures and mitochondria (DiseaseOrCondition) || organism: rats; tissue: longitudinal sections of skeletal muscle; context: old male, hypoxic (HY) group
Maternal hypoxia (DiseaseOrCondition) --AFFECTS-- GSK-3β (ProteinOrGene) || organism: rats; context: old male, hypoxic (HY) group vs. control
Maternal hypoxia (DiseaseOrCondition) --AFFECTS-- GLUT4 (ProteinOrGene) || organism: rats; context: old male, hypoxic (HY) group vs. control
SIRT3/GSK-3β/GLUT4 axis (Pathway) --ASSOCIATED_WITH-- maternal hypoxia-induced skeletal muscle IR (DiseaseOrCondition) || organism: rats; context: old male, hypoxic (HY) group vs. control

End of examples. Please perform the task for the following text.

Text: {request}"""

DEFAULT_PROMPT_BASE_TEMPLATE = "As a biomedical researcher, you are able to extract structured information from a given piece of text. {what_to_extract}"
DEFAULT_PROMPT_ENTITY_TEMPLATE = (
    "The allowed entity types are: {entity_types}."
)
DEFAULT_PROMPT_RELATION_TEMPLATE = "The allowed relation types are: {relation_types}. Relations should only be generated for results and NOT for experiment setup, background or speculative or hypothesized relationships."
DEFAULT_PROMPT_END_TEMPLATE = """The output should look like:
{entity_desc}{relation_desc}{icl}
Do not provide any explanation or deviate from the format. If any entity does not conform to the entity types stated, they should not be included. Please now perform the task for the following text:
{{query}}"""

DEFAULT_PROMPT_END_ENTITY_FORMAT = (
    "Entities:\nEntity1 (EntityType)\nEntity2 (EntityType)\n"
)

DEFAULT_PROMPT_END_RELATION_FORMAT = "\nRelationships:\nEntity1 (EntityType) --RELATIONSHIP_TYPE-- Entity2 (EntityType)\n"

TRIPLEX_PROMPT = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.

**Entity Types:**
{entity_types}

**Predicates:**
{predicates}

**Text:**
{{query}}
"""

PHI_3_INSTRUCT_GRAPH_PROMPT = """
A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

The User provides text in the format:

-------Text begin-------
<User provided text>
-------Text end-------

The Assistant follows the following steps before replying to the User:

1. **identify entities** The Assistant identifies all entities in the text of the types: {entity_types}. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

"nodes":[{{{{\"id\": <entity N>, \"type\": <type>}}}}, ...]

where \"type\": <type> is the type of the entity.

2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the \"nodes\" list defined above. These relationships are called \"edges\" and they follow the structure of:

\"edges\":[{{{{\"from\": <entity 1>, \"to\": <entity 2>, \"label\": <relationship>}}}}, ...]

The <entity N> must correspond to the \"id\" of an entity in the \"nodes\" list{re_description}.

The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
The Assistant responds to the User in JSON only, according to the following JSON schema:

{{{{\"type\":\"object\",\"properties\":{{{{\"nodes\":{{{{\"type\":\"array\",\"items\":{{{{\"type\":\"object\",\"properties\":{{{{\"id\":{{{{\"type\":\"string\"}}}},\"type\":{{{{\"type\":\"string\"}}}},\"detailed_type\":{{{{\"type\":\"string\"}}}}}}}},\"required\":[\"id\",\"type\",\"detailed_type\"],\"additionalProperties\":false}}}}}}}},\"edges\":{{{{\"type\":\"array\",\"items\":{{{{\"type\":\"object\",\"properties\":{{{{\"from\":{{{{\"type\":\"string\"}}}},\"to\":{{{{\"type\":\"string\"}}}},\"label\":{{{{\"type\":\"string\"}}}}}}}},\"required\":[\"from\",\"to\",\"label\"],\"additionalProperties\":false}}}}}}}}}}}},\"required\":[\"nodes\",\"edges\"],\"additionalProperties\":false}}}}

Input:
-------Text begin-------
{{query}}
-------Text end-------
"""

"""
{{{{\"type\":\"object\",\"properties\":{{{{\"nodes\":{{{{\"type\":\"array\",\"items\":{{{{\"type\":\"object\",\"properties\":{{{{\"id\":{{{{\"type\":\"string\"}}}},\"type\":{{{{\"type\":\"string\"}}}},\"detailed_type\":{{{{\"type\":\"string\"}}}},\"required\":[\"id\",\"type\",\"detailed_type\"],\"additionalProperties\":false}}}},\"edges\":{{{{\"type\":\"array\",\"items\":{{{{\"type\":\"object\",\"properties\":{{{{\"from\":{{{{\"type\":\"string\"}}}},\"to\":{{{{\"type\":\"string\"}}}},\"label\":{{{{\"type\":\"string\"}}}},\"required\":[\"from\",\"to\",\"label\"],\"additionalProperties\":false}}}},\"required\":[\"nodes\",\"edges\"],\"additionalProperties\":false}}}}
"""
