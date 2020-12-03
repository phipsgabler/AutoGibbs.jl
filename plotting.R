library(tidyverse)

base_font <- "Linux Biolinum O"
theme_set(theme_bw(
    base_family = base_font,
    base_size = 10))
## theme_update(text = element_text(color = "black"))
theme_update(
    strip.background = element_rect(fill = "white"))

model <- "IMM"
timestamp <- "2020-11-30T12:01:00"
results_dir <- "results/"
inspected_parameters = c("z[5]", "μ[1]", "v[1]")


sampling_times <- read_csv(paste0(results_dir, model, "-sampling_times-", timestamp, ".csv"))
diagnostics <- read_csv(paste0(results_dir, model, "-diagnostics-", timestamp, ".csv")) %>%
    mutate(diagnostic = as.factor(diagnostic))
chains <- read_csv(paste0(results_dir, model, "-chains-", timestamp, ".csv"))
compile_times <- read_csv(paste0(results_dir, model, "-compile_times-", timestamp, ".csv")) %>%
    mutate(is_first = repetition == 1,
           category = as_factor(ifelse(is_first, "First", "Other")))

## fix inconsistent parameter names in HMM; same for `diagnostics`
## library(forcats)
## chains$parameter <-
##     fct_collapse(chains$parameter,
##                  `T[1, 1]` = c("T[1, 1]", "T[1][1]"),
##                  `T[1, 2]` = c("T[1, 2]", "T[1][2]"),
##                  `T[2, 1]` = c("T[2, 1]", "T[2][1]"),
##                  `T[2, 2]` = c("T[2, 2]", "T[2][2]"))
## write_csv(chains, "results/HMM.recoded-chains-2020-12-01T18:05:00.csv")

obs_labeller <- as_labeller(function(obs) paste(obs, "observations"))
particles_labeller <- as_labeller(function(p) paste(p, "particles"))
isfirst_labeller <- as_labeller(function (p) ifelse(p, "First", "Other"))
## diagnostics_labeller <- function (d) ifelse(d == "ess", bquote(ESS), bquote(R))


samplingtime_plot <-
    sampling_times %>%
    arrange(particles) %>%
    ## mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm,
                                       ## ifelse(discrete_algorithm == "PG",
                                              ## str_c(", ", particles, " particles"),
                                              ## "")))) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm))) %>%
    ggplot(aes(x = data_size,
               group = data_size,
               y = sampling_time)) +
    ## geom_boxplot(outlier.alpha = 0.2) +
    ## geom_errorbar(stat = "summary", color = "red") +
    geom_jitter(size = 0.8, height = 0, width = 2) +
    facet_grid(. ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(sampling_times$data_size)) +
    labs(x = "Observations (data size)", y = "Sampling time (s)",
         color = "Observations", title = paste("Sampling times for", model))
ggsave(paste0(results_dir, model, "-sampling_times.pdf"), samplingtime_plot,
       device = cairo_pdf, dpi = "print", units = "cm", width = 7.5, height = 7.5)

rhat_plot <-
    diagnostics %>%
    filter(diagnostic == "r_hat") %>%
    filter(parameter %in% inspected_parameters) %>%
    arrange(particles) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm))) %>%
    ggplot(aes(x = data_size,
               ## color = as_factor(data_size),
               group = factor(data_size),
               y = value)) +
    ## geom_boxplot() + 
    geom_jitter(size = 0.5, height = 0, width = 1) + 
    facet_grid(parameter ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(diagnostics$data_size)) +
    labs(x = "Observations (data size)", y = expression(widehat(R)),
         color = "Observations", title = bquote(widehat(R) ~ "values for" ~ .(model)))
ggsave(paste0(results_dir, model, "-rhat.pdf"), rhat_plot,
       device = cairo_pdf, dpi = "print", units = "cm", width = 7.5, height = 7.5)


ess_plot <-
    diagnostics %>%
    filter(diagnostic == "ess") %>%
    filter(parameter %in% inspected_parameters) %>%
    arrange(particles) %>%
    mutate(algorithm = as_factor(str_c(discrete_algorithm, " & ", continuous_algorithm))) %>%
    ggplot(aes(x = data_size,
               ## color = as_factor(data_size),
               group = factor(data_size),
               y = value)) +
    ## geom_boxplot() + 
    geom_jitter(size = 0.5, height = 0, width = 1) + 
    facet_grid(parameter ~ algorithm, scales = "free_x") +
    scale_x_continuous(breaks = unique(diagnostics$data_size)) +
    labs(x = "Observations (data size)", y = "ESS",
         color = "Observations",
         title = bquote("ESS values for" ~ .(model) * phantom(widehat(R))))
ggsave(paste0(results_dir, model, "-ess.pdf"), ess_plot,
       device = cairo_pdf, dpi = "print", units = "cm", width = 7.5, height = 7.5)



thin_chains <-
    chains %>%
    filter(repetition %in% 1:3, step %% 37 == 0) %>%
    filter(parameter %in% inspected_parameters) %>%
    select(-model, -continuous_algorithm, -particles)

densities_plot <-
    thin_chains %>%
    filter(parameter %in% inspected_parameters) %>% 
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
    filter(repetition == 3) %>%
    ggplot(aes(x = step,
               y = value,
               color = discrete_algorithm)) +
    geom_line(size = 0.6, key_glyph = draw_key_rect) +
    ## geom_density() +
    facet_grid(parameter ~ data_size,
               labeller = labeller(data_size = obs_labeller),
               scales = "free_y") +
    ## guides(color = F) + 
    labs(x = "Step", y = "Sampled value",
         color = "Algorithm",
         title = paste("Chain comparisons for", model))
ggsave(paste0(results_dir, model, "-chains.pdf"), chains_plot,
       device = cairo_pdf, dpi = "print", units = "cm", width = 15.5, height = 7.5)


chains_by_param <- chains %>%
    filter(repetition == 3) %>%
    filter(parameter %in% inspected_parameters) %>%
    select(-model, -continuous_algorithm, -particles) %>%
    group_by(parameter, discrete_algorithm, data_size)
significance_levels <- chains_by_param %>%
    summarize(significance_level = qnorm((1 + 0.95) / 2) / sqrt(sum(!is.na(value))))

acfs <- chains_by_param %>%
    do({
        a <- acf(.$value, plot=F)
        summarize(., lag = a$lag, acf = a$acf)
    })
    

acf_plot <-
    acfs %>%
    ggplot(aes(x = lag, color = discrete_algorithm)) +
    geom_linerange(aes(ymin = 0, ymax = acf, group = discrete_algorithm),
                   size = 0.4,
                   position = position_dodge2(width = 1),
                   key_glyph = draw_key_rect) +
    facet_grid(parameter ~ data_size,
               labeller = labeller(data_size = obs_labeller)) +
    geom_hline(aes(yintercept = significance_level),
               color = "darkgray", size = 0.2,
               data = significance_levels) +
    geom_hline(aes(yintercept = -significance_level),
               color = "darkgray", size = 0.2,
               data = significance_levels) +
    labs(x = 'Lag', y = 'ACF',
         color = "Algorithm",
         title = paste("Autocorrelation estimate for", model))
ggsave(paste0(results_dir, model, "-acfs.pdf"), acf_plot, 
       device = cairo_pdf, dpi = "print", units = "cm", width = 15, height = 7.5)




## filter(compile_times, is_first == F) %>%
compiletime_plot <-
    compile_times %>%
    ggplot(aes(x = data_size,
               y = compilation_time,
               fill = category,
               color = category
               )) +
    geom_jitter(size = 0.8, height = 0, width = 2,
                key_glyph = draw_key_rect) +
    geom_smooth(formula = y ~ I(x^2) - 1, method = "lm", se = FALSE,
                data = filter(compile_times, !is_first),
                color = "black", size = 0.5,
                key_glyph = draw_key_blank) +
    guides(color = FALSE) +
    scale_x_continuous(breaks = unique(sampling_times$data_size)) +
    labs(x = "Observations (data size)", y = "Extraction time (s)",
         fill = "Repetition",
         title = paste("AG extraction times for", model))
ggsave(paste0(results_dir, model, "-compile_times.pdf"), compiletime_plot, 
       device = cairo_pdf, dpi = "print", units = "cm", width = 7.5, height = 7.5)
