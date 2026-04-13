library(BAS)
library(bayesplot)

# Load data
df <- read.csv("data/cleaned/ev_cleaned_for_bayesian.csv")

# Scale all variables
df$EV_stock_s    <- scale(df$EV_stock)
df$sales_share_s <- scale(df$sales_share)
df$stock_share_s <- scale(df$stock_share)
df$EV_sales_s    <- scale(df$EV_sales)

# MODEL 1: Full model — EV_stock + sales_share + stock_share
# Research question: which within-IEA indicators predict sales?
cat("  MODEL 1: Full Model (EV_stock + sales_share + stock_share)\n")

model1 <- bas.lm(
  EV_sales_s ~ EV_stock_s + sales_share_s + stock_share_s,
  data       = df,
  prior      = "JZS",
  modelprior = uniform(),
  method     = "deterministic"
)

summary(model1)
coef1 <- coef(model1)
print(coef1)
confint(coef1, level = 0.95)

cat("\nPosterior R-squared (top model):", round(max(model1$R2), 4), "\n")
cat("Weighted R-squared (BMA)       :", round(sum(model1$postprobs * model1$R2), 4), "\n")

fitted1    <- fitted(model1, estimator = "BMA")
residuals1 <- df$EV_sales_s - fitted1

# MODEL 2: Share-only model — sales_share + stock_share
# Research question: can market penetration indicators alone
# predict EV sales, without relying on cumulative stock?
# This isolates a different signal from Model 1 and provides
# a clearer contrast with Yeh & Wang [3] who use external data.
cat("  MODEL 2: Share-Only Model (sales_share + stock_share)\n")

model2 <- bas.lm(
  EV_sales_s ~ sales_share_s + stock_share_s,
  data       = df,
  prior      = "JZS",
  modelprior = uniform(),
  method     = "deterministic"
)

summary(model2)
coef2 <- coef(model2)
print(coef2)
confint(coef2, level = 0.95)

cat("\nPosterior R-squared (top model):", round(max(model2$R2), 4), "\n")
cat("Weighted R-squared (BMA)       :", round(sum(model2$postprobs * model2$R2), 4), "\n")

fitted2    <- fitted(model2, estimator = "BMA")
residuals2 <- df$EV_sales_s - fitted2

# MODEL COMPARISON SUMMARY
cat("  MODEL COMPARISON\n")
cat(sprintf("  %-35s %10s %10s\n", "Metric", "Model 1", "Model 2"))
cat(sprintf("  %-35s %10s %10s\n", "Predictors", "Full", "Share-only"))
cat(sprintf("  %-35s %10.4f %10.4f\n", "R2 top model", max(model1$R2), max(model2$R2)))
cat(sprintf("  %-35s %10.4f %10.4f\n", "BMA-weighted R2",
            sum(model1$postprobs * model1$R2),
            sum(model2$postprobs * model2$R2)))
cat(sprintf("  %-35s %10.4f %10.4f\n", "RMSE (scaled)",
            sqrt(mean(residuals1^2)),
            sqrt(mean(residuals2^2))))


# PLOTS — MODEL 1
# Posterior coefficients
png("outputs/model1_posterior_coefficients.png", width = 800, height = 600)
plot(coef1, ask = FALSE)
dev.off()

# Fitted vs actual
png("outputs/model1_fitted_vs_actual.png", width = 800, height = 600)
plot(df$EV_sales_s, fitted1,
     xlab = "Actual EV Sales (scaled)",
     ylab = "Fitted EV Sales (scaled)",
     main = "Model 1 (Full): Actual vs Fitted",
     pch  = 16, col = "steelblue")
abline(a = 0, b = 1, col = "red", lty = 2)
dev.off()

# Residuals
png("outputs/model1_residuals.png", width = 800, height = 600)
plot(fitted1, residuals1,
     xlab = "Fitted Values",
     ylab = "Residuals",
     main = "Model 1 (Full): Residuals vs Fitted",
     pch  = 16, col = "purple")
abline(h = 0, col = "black", lty = 2)
dev.off()

# PLOTS — MODEL 2
# Posterior coefficients
png("outputs/model2_posterior_coefficients.png", width = 800, height = 600)
plot(coef2, ask = FALSE)
dev.off()

# Fitted vs actual
png("outputs/model2_fitted_vs_actual.png", width = 800, height = 600)
plot(df$EV_sales_s, fitted2,
     xlab = "Actual EV Sales (scaled)",
     ylab = "Fitted EV Sales (scaled)",
     main = "Model 2 (Share-only): Actual vs Fitted",
     pch  = 16, col = "darkorange")
abline(a = 0, b = 1, col = "red", lty = 2)
dev.off()

# Residuals
png("outputs/model2_residuals.png", width = 800, height = 600)
plot(fitted2, residuals2,
     xlab = "Fitted Values",
     ylab = "Residuals",
     main = "Model 2 (Share-only): Residuals vs Fitted",
     pch  = 16, col = "tomato")
abline(h = 0, col = "black", lty = 2)
dev.off()

# SIDE-BY-SIDE COMPARISON PLOT
png("outputs/model_comparison_fitted.png", width = 1400, height = 600)
par(mfrow = c(1, 2))

plot(df$EV_sales_s, fitted1,
     xlab = "Actual EV Sales (scaled)",
     ylab = "Fitted EV Sales (scaled)",
     main = "Model 1: Full (EV stock + shares)\nR² = 0.9879",
     pch  = 16, col = "steelblue")
abline(a = 0, b = 1, col = "red", lty = 2)

plot(df$EV_sales_s, fitted2,
     xlab = "Actual EV Sales (scaled)",
     ylab = "Fitted EV Sales (scaled)",
     main = "Model 2: Share-only (sales_share + stock_share)",
     pch  = 16, col = "darkorange")
abline(a = 0, b = 1, col = "red", lty = 2)

dev.off()
