library(tidyverse)

base_font <- "Linux Biolinum O"
theme_set(theme_light(base_family = base_font))

sampling_times <- read_csv("results/GMM-sampling_times-2020-10-12T19:21:00.csv")
diagnostics <- read_csv("results/GMM-diagnostics-2020-10-12T19:21:00.csv") %>%
    mutate(diagnostic = as.factor(diagnostic))
chains <- read_csv("results/GMM-chains-2020-10-12T19:21:00.csv")
compile_times <- read_csv("results/GMM-compile_times-2020-10-12T19:21:00.csv") %>%
    mutate(is_first = repetition == 1)

obs_labeller <- as_labeller(function(obs) paste(obs, "observations"))
particles_labeller <- as_labeller(function(p) paste(p, "particles"))
isfirst_labeller <- as_labeller(function (p) ifelse(p, "First", "Other"))
## diagnostics_labeller <- function (d) ifelse(d == "ess", bquote(ESS), bquote(R))


samplingtime_plot_gmm <-
    sampling_times %>%
    arrange(particles) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ifelse(discrete_algorithm == "PG",
                                              str_c(", ", particles, " particles"),
                                              "")))) %>%
    ggplot(aes(x = data_size,
               group = data_size,
               y = sampling_time)) +
    geom_boxplot(outlier.alpha = 0.2) +
    ## geom_errorbar(stat = "summary", color = "red") +
    ## geom_point(alpha = 0.5) +
    facet_grid(. ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(sampling_times$data_size)) +
    labs(x = "Observations (data size)", y = "Sampling time (s)",
         color = "Observations", title = "Sampling times",
         subtitle = "Factored by algorithm and number of PG particles")
ggsave("results/GMM-sampling_times.pdf", samplingtime_plot_gmm, device = cairo_pdf)

diagnostics_plot_gmm <- diagnostics %>%
    filter(diagnostic == "ess", data_size != 25) %>%
    filter(!startsWith(parameter, "z") | parameter == "z[1]") %>%
    arrange(particles) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ifelse(discrete_algorithm == "PG",
                                              str_c(", ", particles, " particles"),
                                              "")))) %>%
    ggplot(aes(x = data_size,
               ## group = interaction(data_size, parameter),
               group = data_size,
               y = value)) +
    ## geom_boxplot(outlier.alpha = 0.2, position = position_dodge2()) +
    ## geom_boxplot(outlier.alpha = 0.2) + 
    geom_jitter(alpha = 0.5) + 
    facet_grid(diagnostic + parameter ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(diagnostics$data_size)) +
    labs(x = "Observations (data size)", y = "Value of diagnostic",
         color = "Observations", title = "Convergence diagnostics",
         subtitle = "Factored by algorithm and number of particles")
ggsave("results/GMM-diagnostics.pdf", diagnostics_plot_gmm, device = cairo_pdf)

chains_plot_gmm <- chains %>%
    filter(discrete_algorithm == "AG", repetition %in% 1:3, step %% 37 == 0) %>%
    filter((!startsWith(parameter, "z") | parameter == "z[1]")) %>%
    select(-model, -discrete_algorithm, -continuous_algorithm, -particles) %>%
    ## arrange(particles) %>%
    ## mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ## ifelse(discrete_algorithm == "PG",
                                              ## str_c(", \n", particles, " particles"),
                                              ## "")))) %>%
    ggplot(aes(x = step,
               y = value,
               color = as_factor(repetition))) +
    geom_line(alpha = 0.5) +
    ## geom_density() +
    facet_grid(parameter ~ data_size,
               labeller = labeller(data_size = obs_labeller),
               scales = "free_y") +
    guides(color = F) + 
    labs(x = "Step", y = "Sampled value",
         color = "Chain", title = "AG chains",
         subtitle = paste("Factored by number of observations (data size)",
                          "and selected parameters"))
ggsave("results/GMM-chains.pdf", chains_plot_gmm, device = cairo_pdf)

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

## filter(compile_times, is_first == F) %>%
compiletime_plot_gmm <- compile_times %>%
    ggplot(aes(x = data_size,
               y = compilation_time,
               shape = as_factor(ifelse(is_first, "First", "Other")))) +
    geom_point(alpha = 0.5, size = 3) +
    geom_smooth(formula = y ~ I(x^2), method = "lm", se = F,
                data = filter(compile_times, is_first == F),
                color = "black") +
    annotate("text", x = 85, y = 100,
             label = expression("Linear fit of" ~ time %~% datasize^2),
             family = base_font) + 
    ## scale_x_continuous(breaks = unique(compile_times$data_size)) +
    labs(x = "Observations (data size)", y = "Extraction time (s)",
         shape = "Repetition",
         title = "Extraction times for AutoGibbs",
         subtitle = paste(
             "Measuring both compilation of the traced code and the conditional calculation.",
             "All 2 or 3 repetitions per data size class are shown."))
ggsave("results/GMM-compile_times.pdf", compiletime_plot_gmm, device = cairo_pdf)
