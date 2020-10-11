library(tidyverse)

sampling_times <- read_csv("results/sampling_times.csv")
diagnostics <- read_csv("results/diagnostics.csv")
chains <- read_csv("results/chains.csv")
compile_times <- read_csv("results/compile_times.csv")

sampling_times %>% filter(repetition > 1, discrete_algorithm == "PG") %>%
    ggplot(aes(x = continuous_algorithm,
               y = sampling_time,
               color =  as.factor(data_size))) +
    ## stat_boxplot(geom = "errorbar") +
    geom_point(position = position_jitterdodge()) +
    facet_grid(. ~ as.factor(particles), scales = "free_y") +
    theme_light() +
    labs(x = "Algorithm", y = "Sampling time (s)",
         color = "Data size", title = "Sampling times for Particle Gibbs",
         subtitle = "Factored by number of particles")

sampling_times %>% filter(repetition > 1, discrete_algorithm == "AG") %>%
    ggplot(aes(x = continuous_algorithm,
               y = sampling_time,
               color =  as.factor(data_size))) +
    ## stat_boxplot(geom = "errorbar") +
    geom_point(position = position_jitterdodge()) +
    ## facet_grid(. ~ as.factor(particles), scales = "free_y") +
    theme_light() +
    labs(x = "Algorithm", y = "Sampling time (s)",
         color = "Data size", title = "Sampling times for AutoGibbs")

diagnostics %>%
    filter(repetition > 1, (diagnostic == "ess" | value < 20), discrete_algorithm == "PG") %>%
    ggplot(aes(x = continuous_algorithm,
               y = value,
               color =  as.factor(data_size))) +
    ## geom_boxplot() +
    geom_point(position = position_jitterdodge()) +
    facet_grid(diagnostic ~ as.factor(particles), scales = "free_y") +
    theme_light() +
    labs(x = "Algorithm", y = "Value",
         color = "Data size", title = "Convergence diagnostics of Particle Gibbs",
         subtitle = expression(paste("Factored by number of particles; ",
                                     hat(R), " outliers over 20 removed for readability")))
