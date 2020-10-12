library(tidyverse)

sampling_times <- read_csv("results/sampling_times.csv") %>%
    mutate(particles = as.factor(particles), data_size = as.factor(data_size))
diagnostics <- read_csv("results/diagnostics.csv") %>%
    mutate(particles = as.factor(particles),
           data_size = as.factor(data_size),
           diagnostic = as.factor(diagnostic))
chains <- read_csv("results/chains.csv") %>%
    mutate(particles = as.factor(particles),
           data_size = as.factor(data_size))
compile_times <- read_csv("results/compile_times.csv") %>%
    mutate(data_size = as.factor(data_size)) %>%
    transmute(is_first = repetition == 1) %>%
    rename(compile_time = sampling_time)

obs_labeller <- as_labeller(function(obs) paste(obs, "observations"))
particles_labeller <- as_labeller(function(p) paste(p, "particles"))
## diagnostics_labeller <- function (d) ifelse(d == "ess", bquote(ESS), bquote(R))


sampling_times %>% filter(repetition > 1, discrete_algorithm == "PG") %>%
    ggplot(aes(x = continuous_algorithm,
               y = sampling_time,
               color =  data_size)) +
    ## stat_boxplot(geom = "errorbar") +
    geom_point(position = position_jitterdodge()) +
    facet_grid(. ~ particles, scales = "free_y", labeller = particles_labeller) +
    theme_light() +
    labs(x = "Algorithm", y = "Sampling time (s)",
         color = "Observations", title = "Sampling times for Particle Gibbs",
         subtitle = "Factored by number of particles")

sampling_times %>% filter(repetition > 1, discrete_algorithm == "AG") %>%
    ggplot(aes(x = continuous_algorithm,
               y = sampling_time,
               color =  data_size)) +
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
               color = data_size)) +
    geom_point(position = position_jitterdodge()) +
    facet_grid(diagnostic ~ particles, scales = "free_y",
               labeller = labeller(particles = particles_labeller)) +
    theme_light() +
    labs(x = "Algorithm", y = "Value",
         color = "Data size", title = "Convergence diagnostics of Particle Gibbs",
         subtitle = expression(paste("Factored by number of particles; ",
                                     hat(R), " outliers over 20 removed for readability")))

chains_pg <- chains %>%
    filter(discrete_algorithm == "PG", (!startsWith(parameter, "z") | parameter == "z[1]")) %>%
    ggplot(aes(x = step,
               y = value,
               color = as.factor(repetition))) +
    geom_line() +
    ## geom_density() +
    facet_grid(parameter ~ continuous_algorithm + data_size + particles,
               labeller = labeller(particles = particles_labeller,
                                   data_size = obs_labeller),
               scales = "free_y") +
    guides(color = F) + 
    theme_light() +
    labs(x = "Step", y = "Value",
         color = "Chain", title = "Chains",
         subtitle = expression(paste("Factored by number of particles; ")))

chains %>% filter(discrete_algorithm == "PG", (!startsWith(parameter, "z") | parameter == "z[1]")) %>%
    ggplot(aes(x = value,
               color = as.factor(repetition))) +
    geom_density() +
    ## geom_density() +
    facet_grid(parameter ~ continuous_algorithm + data_size + particles,
               labeller = labeller(particles = particles_labeller,
                                   data_size = obs_labeller),
               scales = "free_y") +
    guides(color = F) + 
    theme_light() +
    labs(x = "Step", y = "Value",
         color = "Chain", title = "Chains",
         subtitle = expression(paste("Factored by number of particles; ")))

compile_times %>%
    ggplot(aes(x = data_size,
               y = compile_time)) +
    geom_boxplot() + 
    geom_point(position = position_jitter(height = 0, width = 0.05)) +
    ## stat_boxplot(geom = "errorbar") +
    theme_light() +
    labs(x = "Data size", y = "Compilation time (s)",
         title = "Extraction times for AutoGibbs")

