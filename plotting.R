library(tidyverse)

sampling_times <- read_csv("results/sampling_times.csv")
diagnostics <- read_csv("results/diagnostics.csv")
chains <- read_csv("results/chains.csv")
compile_times <- read_csv("results/compile_times.csv")

sampling_times %>% filter(repetition > 1) %>%
    ggplot(aes(x = interaction(discrete_algorithm, continuous_algorithm),
               y = sampling_time,
               color = as.factor(particles))) +
    geom_boxplot() +
    geom_jitter(height = 0)

diagnostics %>% filter(repetition > 1) %>%
    ggplot(aes(x = interaction(discrete_algorithm, continuous_algorithm),
               y = value,
               color = as.factor(particles))) +
    geom_boxplot() +
    geom_jitter(height = 0) +
    facet_wrap(~ diagnostic, scales = "free_y")
