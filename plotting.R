library(tidyverse)

base_font <- "Linux Biolinum O"
theme_set(theme_light(base_family = base_font))

model <- "HMM"
timestamp <- "2020-11-18T19:02:00"
results_dir <- "results/"


sampling_times <- read_csv(paste0(results_dir, model, "-sampling_times-", timestamp, ".csv"))
diagnostics <- read_csv(paste0(results_dir, model, "-diagnostics-", timestamp, ".csv")) %>%
    mutate(diagnostic = as.factor(diagnostic))
chains <- read_csv(paste0(results_dir, model, "-chains-", timestamp, ".csv"))
compile_times <- read_csv(paste0(results_dir, model, "-compile_times-", timestamp, ".csv")) %>%
    mutate(is_first = repetition == 1)

obs_labeller <- as_labeller(function(obs) paste(obs, "observations"))
particles_labeller <- as_labeller(function(p) paste(p, "particles"))
isfirst_labeller <- as_labeller(function (p) ifelse(p, "First", "Other"))
## diagnostics_labeller <- function (d) ifelse(d == "ess", bquote(ESS), bquote(R))


samplingtime_plot <-
    sampling_times %>%
    arrange(particles) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ifelse(discrete_algorithm == "PG",
                                              str_c(", ", particles, " particles"),
                                              "")))) %>%
    ggplot(aes(x = data_size,
               group = data_size,
               y = sampling_time)) +
    ## geom_boxplot(outlier.alpha = 0.2) +
    geom_errorbar(stat = "summary", color = "red") +
    geom_point(alpha = 0.5) +
    facet_grid(. ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(sampling_times$data_size)) +
    labs(x = "Observations (data size)", y = "Sampling time (s)",
         color = "Observations", title = paste("Sampling times for", model))
ggsave(paste0(results_dir, model, "-sampling_times.pdf"), samplingtime_plot, device = cairo_pdf)

diagnostics_plot <-
    diagnostics %>%
    filter(diagnostic == "ess") %>%
    filter(parameter == "s[5]" | parameter == "m[1]") %>%
    arrange(particles) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ifelse(discrete_algorithm == "PG",
                                              str_c(", ", particles, " particles"),
                                              "")))) %>%
    ggplot(aes(x = data_size,
               ## group = interaction(data_size, parameter),
               color = as_factor(data_size),
               y = value)) +
    ## geom_boxplot(outlier.alpha = 0.2, position = position_dodge2()) +
    ## geom_boxplot(outlier.alpha = 0.2) + 
    geom_jitter(alpha = 0.5, height = 0) + 
    facet_grid(diagnostic + parameter ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(diagnostics$data_size)) +
    labs(x = "Observations (data size)", y = "Value of diagnostic",
         color = "Observations", title = paste("Convergence diagnostics for", model))
ggsave(paste0(results_dir, model, "-diagnostics.pdf"), diagnostics_plot, device = cairo_pdf)


thin_chains <-
    chains %>%
    filter(discrete_algorithm == "AG", repetition %in% 1:3, step %% 37 == 0, particles == 10) %>%
    filter(parameter == "s[5]" | parameter == "m[2]") %>%
    select(-model, -discrete_algorithm, -continuous_algorithm, -particles)

densities_plot <-
    thin_chains %>%
    filter(parameter == "s[5]") %>% 
    ## arrange(particles) %>%
    ## mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ## ifelse(discrete_algorithm == "PG",
                                              ## str_c(", \n", particles, " particles"),
                                              ## "")))) %>%
    ggplot(aes(x = value,
               color = as_factor(repetition))) +
    geom_density() +
    facet_grid(parameter ~ data_size,
               labeller = labeller(data_size = obs_labeller),
               scales = "free") +
    guides(color = F) + 
    labs(x = "Value", y = "Probability",
         title = paste("AG posterior densities for", model))
ggsave(paste0(results_dir, model, "-densities.pdf"), densities_plot, device = cairo_pdf)


chains_plot <-
    thin_chains %>%
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


chains_by_param <- chains %>%
    filter(discrete_algorithm == "AG", repetition == 1, particles == 10) %>%
    filter((!startsWith(parameter, "z") | parameter == "z[10]")) %>%
    select(-model, -discrete_algorithm, -continuous_algorithm, -particles) %>%
    group_by(parameter, data_size)
significance_levels <- chains_by_param %>%
    summarize(significance_level = qnorm((1 + 0.95) / 2) / sqrt(sum(!is.na(value))))

acfs <- chains_by_param %>%
    do({
        a <- acf(.$value, plot=F)
        summarize(., lag = a$lag, acf = a$acf)
    })
    

acf_plot <-
    acfs %>%
    ggplot(aes(x = lag, y = acf)) +
    geom_linerange(aes(ymin = 0, ymax = acf)) +
    facet_grid(parameter ~ data_size,
               labeller = labeller(data_size = obs_labeller)) +
    geom_hline(aes(yintercept = significance_level), lty=3,
               data = significance_levels) +
    geom_hline(aes(yintercept = -significance_level), lty=3,
               data = significance_levels) +
    labs(x = 'Lag', y = 'ACF',
         title = paste("Autocorrelation estimate for", model))
ggsave(paste0(results_dir, model, "-acfs.pdf"), acf_plot, device = cairo_pdf)




## filter(compile_times, is_first == F) %>%
compiletime_plot <-
    compile_times %>%
    ggplot(aes(x = data_size,
               y = compilation_time,
               color = as_factor(ifelse(is_first, "First", "Other")))) +
    geom_point(size = 3) +
    geom_smooth(formula = y ~ I(x^2), method = "lm", se = F,
                data = filter(compile_times, is_first == F),
                color = "black") +
    ## annotate("text", x = 40, y = 15,
    ##          label = expression("Linear fit of" ~ time %~% datasize^2),
    ##          family = base_font) + 
    ## scale_x_continuous(breaks = unique(compile_times$data_size)) +
    labs(x = "Observations (data size)", y = "Extraction time (s)",
         shape = "Repetition",
         color = "Repetition",
         title = paste("AutoGibbs extraction times for", model))
ggsave(paste0(results_dir, model, "-compile_times.pdf"), compiletime_plot, device = cairo_pdf)
