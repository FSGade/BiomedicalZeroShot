library(patchwork)
library(tidyverse)
library(ggridges)
library(ggpubr)
library(ggh4x)
library(RColorBrewer)

update_geom_defaults("text", list(colour = "grey20", family = theme_get()$text$family))

CORPORA_DATA = Sys.getenv("CORPORA_DATA", unset = getwd())

clean_bigbio_names <- read_tsv(paste0(CORPORA_DATA, "/bigbio_names_usedsplits_w_cnt.tsv"))
bigbio_gs_output_cnts <- read_tsv(paste0(CORPORA_DATA, "/bigbio_gs_output_cnts.tsv"))

bigbio_means <- bigbio_gs_output_cnts %>%
  group_by(config_name) %>%
  summarise(mean_query_length = mean(text_length),
            mean_entity_count = mean(entity_count),
            n = n()) %>%
  mutate(query_group = case_when(mean_query_length > 3000 ~ ">3000 chars",
                               mean_query_length > 1500 ~ ">1500 chars",
                               mean_query_length > 500 ~ ">500 chars",
                               TRUE ~ "<=500 chars"),
         query_group = factor(query_group, levels=c("<=500 chars",
                                                    ">500 chars",
                                                    ">1500 chars",
                                                    ">3000 chars"))) %>%
  left_join(., clean_bigbio_names, by=c("config_name")) %>%
  left_join(.,bigbio_gs_output_cnts %>%
              select(config_name, split) %>%
              distinct(), by=c("config_name")) %>%
  mutate(is_over_512 = case_when(count > 512 ~ "*", TRUE ~ ""),
         is_validation = case_when(split == "validation" ~ "†", TRUE ~ "")) %>%
  mutate(plot_name = paste0(display_name, is_validation, " (n=", pmin(count, 512), is_over_512, ")"),
         entity_type_count = paste0(entity_type_count, "   "),
         relation_type_count = case_when(is.na(relation_type_count) ~ NA,
                                         TRUE ~ paste0(relation_type_count, "   "))) %>%
  arrange(desc(mean_query_length)) %>%
  filter(!grepl("jnlpba", config_name))

bigbio_gs_output_cnts = bigbio_gs_output_cnts %>%
  group_by(config_name) %>%
  filter(row_number() <= 512) %>%
  ungroup() %>%
  left_join(., bigbio_means, by="config_name") %>%
  filter(!grepl("jnlpba", config_name)) %>%
  mutate(config_name = factor(config_name, levels=bigbio_means$config_name),
         plot_name = factor(plot_name, levels=bigbio_means$plot_name)) %>%
  mutate(relationship_count = case_when(task == "NER" ~ -1,
                                        TRUE ~ relationship_count))

lancet_w_black = get_palette("lancet", 3)
lancet_w_black = c("#000000FF", lancet_w_black)

input_plots = ggplot(bigbio_gs_output_cnts,
       aes(x = text_length,
           y = plot_name,
           #y = config_name,
           fill = task,
           color = language
       )) +
  geom_boxplot(alpha=0.5, lwd=.2, outlier.shape = 20, outlier.size = .4, outlier.stroke = 0) +
  #geom_density_ridges(alpha=0.25) +
  scale_x_continuous(expand = c(0, 0),
                     limits = c(0, NA)) +
  ggh4x::facet_grid2(query_group ~ "Text length (abstract/text)", scales ="free", space="free_y", independent = "x") +
  theme_pubr(base_size=5) +
  labs(x="Text length (abstract/text)",
       y="Dataset",
       fill="Task",
       color="Language"
       ) +
  theme(
    strip.text = element_text(size=rel(1.2)),
    strip.background.x = element_rect(linewidth=.2),
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    axis.title.y = element_blank(),
    axis.line = element_line(colour = 'black', size = 0.2)
  ) + fill_palette("lancet") + color_palette(lancet_w_black)

entity_plots = ggplot(bigbio_gs_output_cnts,
       aes(x = entity_count,
           y = plot_name,
           fill = task,
           color = language
           )) +
  geom_boxplot(alpha=0.5, lwd=.2, outlier.shape = 20, outlier.size = .4, outlier.stroke = 0) +
  geom_text(aes(
    y=plot_name,
    x=Inf,
    label=entity_type_count
  ), data=bigbio_means, inherit.aes = F, hjust=1,
  size=1.5) +
  #geom_density_ridges(alpha=0.25) +
  scale_x_continuous(expand = c(0, 0),
                     limits = c(0, NA)) +
  ggh4x::facet_grid2(query_group ~ "Entity counts (& unique type count)", scales ="free", space="free_y", independent = "x") +
  theme_pubr(base_size=5) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  labs(x="Entity counts (& unique type count)",
       fill="Task",
       color="Language"
       ) +
  theme(
    strip.text = element_text(size=rel(1.2)),
    strip.background.x = element_rect(linewidth=.2),
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    axis.line = element_line(colour = 'black', size = 0.2)
  ) +
  fill_palette("lancet") +
  color_palette(lancet_w_black)


relation_plots = ggplot(bigbio_gs_output_cnts,
                         aes(x = relationship_count,
                             y = plot_name,
                             fill = task,
                             color = language
                         )) +
  geom_boxplot(alpha=.5, lwd=.2, outlier.shape=20, outlier.size=.4, outlier.stroke=0) +
  geom_text(aes(
    y=plot_name,
    x=Inf,
    label=relation_type_count
  ), data=bigbio_means, inherit.aes = F, hjust=1,
  size=1.5) +
  #geom_density_ridges(alpha=0.25) +
  scale_x_continuous(expand = c(0, 0),
                     limits = c(0, NA)) +
  ggh4x::facet_grid2(query_group ~ "Relationship counts (& unique type count)", scales ="free", space="free_y", independent = "x") +
  theme_pubr(base_size=5) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        strip.background = element_rect(linewidth=.2),
        strip.text = element_text(size=rel(1.2)),
        axis.line = element_line(colour = 'black', size = 0.2)) +
  guides(colour = "none", fill = "none") +
  labs(x="Relationship counts (& unique type count)",
       fill="Task",
       color="Language"
  ) +
  fill_palette(get_palette("lancet", 3)[2:3]) +
  color_palette(lancet_w_black)

input_plots +
  entity_plots +
  relation_plots +
  plot_layout(guides = 'collect', nrow = 1) & theme(legend.position='bottom')

ggsave("bigbio_summary.pdf", width=18, height=20, units="cm")




benchmark_summary <- read_tsv("benchmark_summary.tsv") %>%
  mutate(model_pretty_name = case_when(model_identifier == "gemini_1_5_pro" ~ "Gemini 1.5 Pro",
                                       model_identifier == "scilitllm1_5_14b" ~ "SciLitLLM 1.5 14B",
                                       model_identifier == "gliner_multitask" ~ "GLiNER Multi-task v1.0",
                                       model_identifier == "gliner_large_bio" ~ "GLiNER Large Bio v0.1",
                                       model_identifier == "gliner_medium" ~ "GLiNER Medium v2.5",
                                       model_identifier == "gliner_large" ~ "GLiNER Large v2.5",
                                       model_identifier == "gliner_multitask_large" ~ "GLiNER Multi-task Large v0.5",
                                       model_identifier == "gliner_nuner_zero_4k" ~ "NuNER Zero 4K",
                                       model_identifier == "ZWK_InstructUIE" ~ "InstructUIE",
                                       model_identifier == "zeroshotbioner" ~ "ZeroShotBioNER",
                                       TRUE ~ model_identifier
  ))
benchmark_detailed_summary <- read_tsv("benchmark_detailed_summary.tsv") %>%
  mutate(model_pretty_name = case_when(model_identifier == "gemini_1_5_pro" ~ "Gemini 1.5 Pro",
                                       model_identifier == "scilitllm1_5_14b" ~ "SciLitLLM 1.5 14B",
                                       model_identifier == "gliner_multitask" ~ "GLiNER Multi-task v1.0",
                                       model_identifier == "gliner_large_bio" ~ "GLiNER Large Bio v0.1",
                                       model_identifier == "gliner_medium" ~ "GLiNER Medium v2.5",
                                       model_identifier == "gliner_large" ~ "GLiNER Large v2.5",
                                       model_identifier == "gliner_multitask_large" ~ "GLiNER Multi-task Large v0.5",
                                       model_identifier == "gliner_nuner_zero_4k" ~ "NuNER Zero 4K",
                                       model_identifier == "ZWK_InstructUIE" ~ "InstructUIE",
                                       model_identifier == "zeroshotbioner" ~ "ZeroShotBioNER",
                                       TRUE ~ model_identifier)) %>%
  filter(support != 0)
benchmark_summary_by_type <- read_tsv("benchmark_summary_by_type.tsv")



benchmark_summary %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  left_join(., bigbio_means, by="config_name") %>%
  ggplot(mapping = aes(x=mean_query_length, y=ner_f1, color=model_identifier)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  scale_x_continuous(trans='log10')




benchmark_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  left_join(., bigbio_means, by="config_name") %>%
  ggplot(mapping = aes(x=model_identifier, y=ner_f1, color=query_group)) +
  geom_boxplot()


model_factor_order = benchmark_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>% group_by(model_pretty_name) %>% summarise(ner_f1 = mean(ner_f1)) %>% arrange(-ner_f1) %>% pull(model_pretty_name)


NER_by_text <- benchmark_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  left_join(., bigbio_means, by="config_name") %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order)) %>%
  ggplot(mapping = aes(x=model_pretty_name, y=ner_f1, fill=query_group)) +
  geom_boxplot() +
  scale_y_continuous(expand = c(0,0), limits = c(0,1)) +
  theme_pubr(base_size=18) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size=14)) +
  fill_palette(get_palette("lancet", 4)) +
  labs(y="NER F1", x="Model", fill="Average corpus text length")

NER_by_text

ggsave("NER_performance_by_text_length.pdf", height = 8, width=18)


# benchmark_summary %>%
#   filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
#   filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
#   filter(subanalysis == "-") %>%
#   left_join(., bigbio_means, by="config_name") %>%
#   mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order)) %>%
#   ggplot(mapping = aes(x=model_pretty_name, y=ner_f1, fill=language)) +
#   geom_boxplot() +
#   scale_y_continuous(expand = c(0,0)) +
#   theme_pubr(base_size=10) +
#   fill_palette(get_palette("lancet", 4))



model_factor_order_re = benchmark_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>% group_by(model_pretty_name) %>% summarise(re_f1 = mean(re_f1)) %>% arrange(-re_f1) %>% pull(model_pretty_name)



RTE_by_text <- benchmark_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(!(model_identifier %in% c("gliner_nuner_zero_4k", "gliner_medium", "gliner_large", "zeroshotbioner", "gliner_large_bio"))) %>%
  filter(subanalysis == "-") %>%
  left_join(., bigbio_means, by="config_name") %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order_re)) %>%
  ggplot(mapping = aes(x=model_pretty_name, y=re_f1, fill=query_group)) +
  geom_boxplot() +
  scale_y_continuous(expand = c(0,0), limits = c(0,1)) +
  theme_pubr(base_size=18) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size=14)) +
  fill_palette(get_palette("lancet", 4)) +
  labs(y="RTE F1", x="Model", fill="Average corpus text length")

RTE_by_text

ggsave("RTE_performance_by_text_length.pdf", height = 8, width=18)

NER_by_text / RTE_by_text + plot_layout(guides = "collect", axis_titles = "collect") & theme(legend.position = 'top')

ggsave("NER_RTE_performance_by_text_length.pdf", height = 15, width=18)


#benchmark_detailed_summary %>% filter(task == "ner") %>% select(type) %>% distinct() %>%
#  mutate(group="") %>% write_tsv(., "entity_type_groups.tsv")

entity_type_groups = read_tsv("entity_type_groups.tsv")

benchmark_detailed_summary %>%
  filter(!(config_name %in% c("cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  #filter((config_name %in% c("biored_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  filter(task == "ner") %>%
  left_join(., bigbio_means, by="config_name") %>%
  left_join(., entity_type_groups, by="type") %>%
  select(type, group) %>%
  distinct() %>%
  group_by(group) %>%
  arrange(group, type) %>%
  mutate(types = paste0(type, collapse = ", ")) %>%
  select(group, types) %>%
  distinct() %>% View()

benchmark_detailed_summary %>%
  filter(!(config_name %in% c("jnlpba_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  #filter((config_name %in% c("biored_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  filter(task == "ner") %>%
  left_join(., bigbio_means, by="config_name") %>%
  left_join(., entity_type_groups, by="type") %>%
  select(type, group) %>%
  distinct() %>%
  count(group)


group_order_ner = benchmark_detailed_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  filter(task == "ner") %>%
  left_join(., bigbio_means, by="config_name") %>%
  left_join(., entity_type_groups, by="type") %>%
  group_by(group) %>%
  summarise(f1=mean(f1))%>% arrange(-f1) %>% pull(group)


benchmark_detailed_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  filter(task == "ner") %>%
  left_join(., bigbio_means, by="config_name") %>%
  left_join(., entity_type_groups, by="type") %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order)) %>%
  mutate(group = factor(group, levels = group_order_ner))  %>%
  group_by(type, model_pretty_name) %>%
  mutate(weighted_f1 = weighted.mean(f1, support)) %>%
  distinct(weighted_f1, group, model_pretty_name) %>%
  ggplot(mapping = aes(x=group, y=weighted_f1, fill=model_pretty_name)) +
  geom_boxplot() +
  #stat_summary(geom = "crossbar", fun = "mean", linetype = "dotted", width = .75) +
  scale_y_continuous(expand = c(0,0)) +
  labs(y="F1", x="Entity type group", fill="Model") +
  theme_pubr(base_size=18) +
  fill_palette(get_palette("lancet", 10))

ggsave("NER_performance_by_group_weighted_f1.pdf", height = 8, width=18)

benchmark_detailed_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  filter(task == "ner") %>%
  left_join(., bigbio_means, by="config_name") %>%
  left_join(., entity_type_groups, by="type") %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order)) %>%
  mutate(group = factor(group, levels = group_order_ner)) %>%
  ggplot(mapping = aes(x=group, y=f1, fill=model_pretty_name)) +
  geom_boxplot() +
  #stat_summary(geom = "crossbar", fun = "mean", linetype=11, color="white",
  #             width = 0.75, position=position_dodge(),
  #             mapping=aes(group=interaction(group,model_pretty_name))) +
  scale_y_continuous(expand = c(0,0)) +
  labs(y="F1", x="Entity type group", fill="Model") +
  theme_pubr(base_size=18) +
  fill_palette(get_palette("lancet", 10))

ggsave("NER_performance_by_group.pdf", height = 8, width=18)

set.seed(42)
benchmark_summary_main_with_rank = benchmark_summary %>%
  filter(!(config_name %in% c("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"))) %>%
  filter(!(model_identifier %in% c("deepseek_r1_distill_qwen_14b", "meta_llama3_1_405b_instruct_v1_0", "meta_llama3_1_8b_instruct_v1_0", "qwen2_5_14b", "triplex", "scilitllm1_5_7b", "phi3_mini_4k_graph"))) %>%
  filter(subanalysis == "-") %>%
  left_join(., bigbio_means, by="config_name") %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = rev(model_factor_order))) %>%
  group_by(config_name) %>%
  mutate(rank_f1 = rank(-ner_f1, ties.method=c("random"))) %>%
  ungroup()

benchmark_summary_main_with_rank %>%
  ggplot(mapping = aes(x=rank_f1, y=model_pretty_name, fill=model_pretty_name)) +
  geom_density_ridges(quantile_lines = TRUE,
                      position = "points_sina", quantiles = 2, alpha=0.5, bandwidth=0.3,
                      #jittered_points = TRUE,
                      scale = 1) +
  theme_pubr(base_size=12) +
  scale_x_continuous(limits = c(1,10)) +
  fill_palette(rev(get_palette("lancet", 10)))


model_pretty_name_w_rank_tbl <- benchmark_summary_main_with_rank %>%
  group_by(model_pretty_name) %>%
  summarise(median_rank = median(rank_f1),
            median_f1 = median(ner_f1)) %>%
  arrange(median_rank, -median_f1) %>%
  mutate(model_pretty_name_w_rank = paste0(model_pretty_name, " (", median_rank, ")")) %>%
  select(model_pretty_name, model_pretty_name_w_rank)


model_rank_factor_order = model_pretty_name_w_rank_tbl %>% pull(model_pretty_name_w_rank)

benchmark_summary_main_with_rank %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order)) %>%
  left_join(., model_pretty_name_w_rank_tbl, by="model_pretty_name") %>%
  mutate(model_pretty_name_w_rank = factor(model_pretty_name_w_rank, levels = model_rank_factor_order)) %>%
  ggplot(mapping = aes(x=rank_f1, fill=model_pretty_name)) +
  geom_bar(width=1) +
  scale_x_continuous(limits = c(0.5,10.5), expand=c(0,0), breaks=seq(1,10)) +
  scale_y_continuous(limits = c(0,NA), expand=c(0,0), breaks=c(0,10,20)) +
  facet_grid(rows=vars(model_pretty_name)) +
  theme_pubr(base_size=8) +
  fill_palette(get_palette("lancet", 10)) +
  labs(x="NER F1 Rank", y="", fill="Model (median rank)") +
  theme(
    strip.background = element_blank(),
    strip.text.y = element_blank()
  )


ggsave("NER_rank_by_model_ridges.pdf", height = 10, width=8)



benchmark_summary_main_with_rank %>%
  mutate(model_pretty_name = factor(model_pretty_name, levels = model_factor_order)) %>%
  left_join(., model_pretty_name_w_rank_tbl, by="model_pretty_name") %>%
  mutate(model_pretty_name_w_rank = factor(model_pretty_name_w_rank, levels = model_rank_factor_order)) %>%
  ggplot(mapping = aes(x=rank_f1, fill=model_pretty_name_w_rank)) +
  geom_bar(width=1, colour = "NA") +
  scale_x_continuous(limits = c(0.5,10.5), expand=c(0,0), breaks=seq(1,10)) +
  scale_y_continuous(limits = c(0,NA), expand=c(0,0), breaks=seq(0,100,10)) +
  theme_pubr(base_size=10, legend = "right") +
  fill_palette(get_palette("lancet", 10)) +
  labs(x="NER F1 Rank", y="", fill="Model (median rank)")

ggsave("NER_rank_by_model.pdf", height = 6, width=8)

benchmark_summary_main_with_rank %>%
  filter(model_identifier == "ZWK_InstructUIE",
         rank_f1 == 1)


benchmark_detailed_summary %>%
  filter(model_identifier == "ZWK_InstructUIE",
         config_name %in% c("citation_gia_test_collection_bigbio_kb","iepa_bigbio_kb","genetaggold_bigbio_kb")) %>%
  select(config_name, type, f1, precision,recall)


zsbn_configs = benchmark_summary_main_with_rank %>%
  filter(model_identifier == "zeroshotbioner",
         rank_f1 == 1) %>%
  pull(config_name)

benchmark_detailed_summary %>%
  filter(model_identifier == "zeroshotbioner",
         config_name %in% !!zsbn_configs,
         task=="ner") %>%
  select(config_name, type, f1, precision,recall)




win_rate <- as_tibble(t(combn(unique(benchmark_summary_main_with_rank$model_identifier), 2)))
colnames(win_rate) = c("model.x","model.y")

set.seed(42)
win_rate = win_rate %>%
  left_join(., benchmark_summary_main_with_rank %>%
              select(model_identifier, config_name, ner_f1),
            by=c("model.x" = "model_identifier"),
            relationship = "many-to-many") %>%
  left_join(., benchmark_summary_main_with_rank %>%
              select(model_identifier, config_name, ner_f1),
            by=c("model.y" = "model_identifier", "config_name" = "config_name")) %>%
  mutate(winner = case_when(ner_f1.x > ner_f1.y ~ model.x,
                            ner_f1.x != ner_f1.y ~ model.y,
                            runif(1) < 0.5 ~ model.x,
                            TRUE ~ model.y))

model_ids = benchmark_summary_main_with_rank %>% distinct(model_identifier) %>% pull

win_pcts = c()

for (model_id in model_ids) {
  #message(model_id)
  win_pcts = c(win_pcts, win_rate %>%
  filter(model.x == !!model_id | model.y == !!model_id) %>%
  mutate(is_winner = (winner == !!model_id)) %>%
  summarise(win_pct = round(mean(is_winner)*100, 1)) %>% pull(win_pct)
  )
}

win_pcts_tbl <- tibble(model_identifier = model_ids, win_pct = win_pcts) %>% arrange(-win_pct)
win_pcts_tbl




win_rate_re <- as_tibble(t(combn(unique(benchmark_summary_main_with_rank$model_identifier), 2)))
colnames(win_rate_re) = c("model.x","model.y")

set.seed(42)
win_rate_re = win_rate_re %>%
  filter(!(model.x %in% c("gliner_nuner_zero_4k", "gliner_medium", "gliner_large", "zeroshotbioner", "gliner_large_bio"))) %>%
  filter(!(model.y %in% c("gliner_nuner_zero_4k", "gliner_medium", "gliner_large", "zeroshotbioner", "gliner_large_bio"))) %>%
  left_join(., benchmark_summary_main_with_rank,
            by=c("model.x" = "model_identifier"),
            relationship = "many-to-many") %>%
  select(model.x, model.y, config_name, re_f1, task) %>%
  filter(task == "NER & RE") %>%
  left_join(., benchmark_summary_main_with_rank %>%
              select(model_identifier, config_name, re_f1),
            by=c("model.y" = "model_identifier", "config_name" = "config_name")) %>%
  mutate(winner = case_when(re_f1.x > re_f1.y ~ model.x,
                            re_f1.x != re_f1.y ~ model.y,
                            runif(1) < 0.5 ~ model.x,
                            TRUE ~ model.y))

model_ids_re = benchmark_summary_main_with_rank %>%
  filter(!(model_identifier %in% c("gliner_nuner_zero_4k", "gliner_medium", "gliner_large", "zeroshotbioner", "gliner_large_bio"))) %>%
  distinct(model_identifier) %>% pull

win_pcts_re = c()

for (model_id in model_ids_re) {
  #message(model_id)
  win_pcts_re = c(win_pcts_re, win_rate_re %>%
                 filter(model.x == !!model_id | model.y == !!model_id) %>%
                 mutate(is_winner = (winner == !!model_id)) %>%
                 summarise(win_pct = round(mean(is_winner)*100, 1)) %>% pull(win_pct)
  )
}

win_pcts_tbl_re <- tibble(model_identifier = model_ids_re, win_pct = win_pcts_re) %>% arrange(-win_pct)
win_pcts_tbl_re

win_pcts_tbl_re %>% pull(win_pct) %>% sum
