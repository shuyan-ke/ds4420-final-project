library(BAS)
library(bayesplot)

# Load data
df <- read.csv("data/cleaned/ev_cleaned_for_bayesian.csv")

# Scale predictors — important for Bayesian priors to be meaningful
df$EV_stock_s    <- scale(df$EV_stock)
df$sales_share_s <- scale(df$sales_share)
df$stock_share_s <- scale(df$stock_share)
df$EV_sales_s    <- scale(df$EV_sales)

# Fit Bayesian linear regression using BAS
# prior = "JZS" (Jeffreys-Zellner-Siow) — weakly informative, standard choice
model <- bas.lm(
  EV_sales_s ~ EV_stock_s + sales_share_s + stock_share_s,
  data         = df,
  prior        = "JZS",
  modelprior   = uniform(),
  method       = "deterministic"
)

# Coefficient estimates — posterior means and credible intervals
summary(model)
coef_model <- coef(model)
print(coef_model)
confint(coef_model, level = 0.95)

# R-squared — posterior mean under BMA
r2_vals <- model$R2
cat("Posterior R-squared (top model):", round(max(r2_vals), 4), "\n")
cat("Weighted R-squared (BMA)       :", round(sum(model$postprobs * r2_vals), 4), "\n")

# Fitted vs actual
fitted_vals <- fitted(model, estimator = "BMA")  # Bayesian Model Averaging

# Posterior predictive check — residuals
residuals_vals <- df$EV_sales_s - fitted_vals

png("outputs/posterior_coefficients.png", width = 800, height = 600)
plot(coef_model, ask = FALSE)
dev.off()

png("outputs/r_squared.png", width = 800, height = 600)
plot(model, which = 1, main = "Posterior R-squared")
dev.off()

png("outputs/fitted_vs_actual.png", width = 800, height = 600)
plot(df$EV_sales_s, fitted_vals,
     xlab = "Actual EV Sales (scaled)",
     ylab = "Fitted EV Sales (scaled)",
     main = "Actual vs Fitted — Bayesian Linear Regression (BAS)",
     pch  = 16, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)
dev.off()

png("outputs/residuals.png", width = 800, height = 600)
plot(fitted_vals, residuals_vals,
     xlab = "Fitted Values",
     ylab = "Residuals",
     main = "Residuals vs Fitted",
     pch  = 16, col = "purple")
abline(h = 0, col = "orange", lty = 2)
dev.off()
