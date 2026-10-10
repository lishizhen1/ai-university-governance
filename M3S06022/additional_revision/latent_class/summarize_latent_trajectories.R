# Aggregate reporting for formal lcmm::hlme fits; no mortality endpoint is loaded.
args <- commandArgs(trailingOnly=TRUE)
root <- gsub('\\\\','/',args[1])
library(lcmm)
library(jsonlite)
private <- file.path(root,'private_local_only')
variant <- if(length(args)>=2) args[2] else 'observed'
stopifnot(variant %in% c('observed','complete'))
train <- read.csv(file.path(private,paste0('mimic_',variant,'_long.csv')))
external <- read.csv(file.path(private,paste0('eicu_',variant,'_long.csv')))
if(variant=='complete') {
 root <- file.path(root,'complete_case')
 private <- file.path(root,'private_local_only')
}
train$id <- as.numeric(train$id); external$id <- as.numeric(external$id)
diags <- read.csv(file.path(root,'model_initialization_diagnostics.csv'))
fits <- list()
for(k in 1:4) {
 p <- file.path(private,'models',paste0('gaussian_RI_K',k,'.rds'))
 if(file.exists(p)) fits[[as.character(k)]] <- readRDS(p)
}
stopifnot(length(fits)>=1)
models <- list(); class_rows <- list(); quality_rows <- list(); profile_rows <- list()
private_trajectories <- list(); posterior_tables <- list()
for(label in names(fits)) {
 m <- fits[[label]]; k <- m$ng
 stopifnot(m$conv==1,m$ns==length(unique(train$id)),
           identical(as.numeric(m$pred$obs),as.numeric(train$sic)),
           identical(as.numeric(m$pred$id),as.numeric(train$id)))
 yhat <- predictY(m,newdata=data.frame(time=0:2),var.time='time',draws=FALSE)$pred
 yhat <- as.matrix(yhat)
 stopifnot(ncol(yhat)==k,nrow(yhat)==3)
 # Neutral labels sorted by model-estimated W1 level only, never by an outcome.
 ordering <- order(yhat[1,],yhat[3,]-yhat[1,])
 class_map <- match(seq_len(k),ordering)
 V <- matrix(0,nrow=length(m$best),ncol=length(m$best))
 V[upper.tri(V,diag=TRUE)] <- m$V
 V <- V+t(V)-diag(diag(V))
 ev <- tryCatch(eigen(V,symmetric=TRUE,only.values=TRUE)$values,error=function(e)NA_real_)
 full <- subset(diags,k==m$ng & phase=='full' & conv==1)
 short <- subset(diags,k==m$ng & phase=='short')
 repeat_ll <- if(nrow(full)) sum(abs(full$loglik-m$loglik)<=1e-4) else 0L
 models[[label]] <- data.frame(k=k,n_subjects=m$ns,n_measurements=nrow(train),parameters=length(m$best),
 loglik=m$loglik,AIC=m$AIC,BIC=m$BIC,conv=m$conv,niter=m$niter,
 gconv_parameters=m$gconv[1],gconv_likelihood=m$gconv[2],gconv_derivatives=m$gconv[3],
 short_initializations=nrow(short),short_converged_starts=sum(short$conv==1),
 short_best_loglik_repeats_tolerance_1e_4=sum(short$conv==1 & abs(short$loglik-m$loglik)<=1e-4),
 full_converged_starts=nrow(full),best_loglik_repeats_tolerance_1e_4=repeat_ll,
 parameter_covariance_min_eigenvalue=min(ev),parameter_covariance_condition=max(ev)/min(ev),
 random_intercept_variance_near_zero=any(abs(m$best[grepl('varcov',names(m$best))])<1e-6),
 residual_sd_near_zero=any(abs(m$best[names(m$best)=='stderr'])<1e-6))
 private_trajectories[[label]] <- data.frame(k=k,class=rep(seq_len(k),each=3),time=rep(0:2,k),
 predicted_SIC=as.vector(yhat[,ordering,drop=FALSE]))
 for(dataset in c('MIMIC-IV','eICU')) {
  raw <- if(dataset=='MIMIC-IV') train else external
  pp <- if(dataset=='MIMIC-IV') m$pprob else predictClass(m,newdata=raw,subject='id')
  probs <- as.matrix(pp[,paste0('prob',ordering),drop=FALSE])
  stopifnot(nrow(pp)==length(unique(raw$id)),all(is.finite(probs)),max(abs(rowSums(probs)-1))<1e-6)
  assignment <- max.col(probs,ties.method='first')
  maxp <- apply(probs,1,max)
  entropy <- if(k>1) 1+sum(ifelse(probs>0,probs*log(probs),0))/(nrow(probs)*log(k)) else NA_real_
  quality_rows[[length(quality_rows)+1]] <- data.frame(k=k,dataset=dataset,n=nrow(pp),entropy=entropy,
   mean_max_posterior=mean(maxp),median_max_posterior=median(maxp),
   proportion_max_posterior_lt_0_7=mean(maxp<0.7),proportion_max_posterior_lt_0_8=mean(maxp<0.8),
   min_assigned_class_n=min(tabulate(assignment,nbins=k)),min_assigned_class_pct=100*min(tabulate(assignment,nbins=k))/nrow(pp))
  ptable <- data.frame(id=pp$id,class=assignment,probs,check.names=FALSE)
  names(ptable)[-(1:2)] <- paste0('prob',seq_len(k))
  write.csv(ptable,file.path(private,paste0('posterior_',gsub('-','',dataset),'_K',k,'.csv')),row.names=FALSE)
  joined <- merge(raw,ptable[,c('id','class')],by='id')
  for(g in seq_len(k)) {
   idx <- assignment==g
   class_rows[[length(class_rows)+1]] <- data.frame(k=k,dataset=dataset,class=g,assigned_n=sum(idx),
    assigned_pct=100*mean(idx),posterior_expected_n=sum(probs[,g]),
    average_posterior_assigned=if(any(idx)) mean(probs[idx,g]) else NA_real_,
    pct_assigned_posterior_ge_0_7=if(any(idx))100*mean(probs[idx,g]>=0.7) else NA_real_,
    pct_assigned_posterior_ge_0_8=if(any(idx))100*mean(probs[idx,g]>=0.8) else NA_real_)
   for(t in 0:2) {
    values <- joined$sic[joined$class==g & joined$time==t]
    profile_rows[[length(profile_rows)+1]] <- data.frame(k=k,dataset=dataset,class=g,time=t,n=length(values),
     observed_mean=if(length(values))mean(values) else NA_real_,observed_sd=if(length(values)>1)sd(values) else NA_real_)
   }
  }
 }
}
model_table <- do.call(rbind,models)
model_table$candidate_set_complete <- all(1:4 %in% model_table$k)
model_table$best_BIC_among_tested_converged <- model_table$BIC==min(model_table$BIC)
write.csv(model_table,file.path(root,'model_comparison.csv'),row.names=FALSE)
write.csv(do.call(rbind,class_rows),file.path(root,'class_summary.csv'),row.names=FALSE)
write.csv(do.call(rbind,quality_rows),file.path(root,'classification_quality.csv'),row.names=FALSE)
write.csv(do.call(rbind,profile_rows),file.path(root,'class_observed_profiles.csv'),row.names=FALSE)
write.csv(do.call(rbind,private_trajectories),file.path(private,'fitted_class_trajectories.csv'),row.names=FALSE)
write_json(list(R=as.character(getRversion()),lcmm=as.character(packageVersion('lcmm')),data_variant=variant,
 model='Gaussian latent class linear mixed model: SIC ~ time; class-specific intercept and time slope; subject random intercept; common random-intercept variance and residual variance; constant class probabilities.',
 fixed='sic ~ time',mixture='~ time',random='~ 1',classmb='~ 1',time_unit='W1=0,W2=1,W3=2; one step = 24 hours',
 candidate_classes=1:4,missingness=if(variant=='observed') 'Observed SIC values only; missing later outcomes omitted from likelihood. No W1 carry or other imputation. Likelihood interpretation requires missing at random conditional on the model; informative observation is unresolved.' else 'Only admissions with all three SIC measurements observed before any carry/imputation are included. Complete-case selection may be informative; comparison with full-cohort results changes the analysis population.',
 initialization='For each K>1, 40 seeded random initialisations from the K1 model, each up to 30 iterations; six highest short-run likelihood candidates fully refitted to maxiter=1000. All convergence thresholds 1e-5; only conv=1 models used.',
 class_selection='Minimum BIC among converged K1-K4 models, supported by AIC, likelihood repetition, class sizes and posterior quality; no mortality endpoint used.',
 covariance_diagnostics='Eigenvalue and condition summaries refer to lcmm parameter covariance on its internal scale, including Cholesky-transformed variance parameters; they are not the random-effects covariance matrix condition number.',
 external='Frozen MIMIC model posterior allocation to observed eICU SIC values, reported descriptively. No external re-selection, model refitting or mortality predictive validation. eICU admissions include repeated persons.',
 limitations=c('SIC is bounded and ordered from 0 to 6; Gaussian likelihood is a continuous approximation and latent classes may approximate a non-Gaussian distribution.',
 'Only three 24-hour windows are available; no higher-order trajectory is estimated.',
 if(variant=='observed') '1,341 of 10,810 MIMIC admissions (12.4%) have one observed SIC value; these inform level but not individual change.' else 'Complete-case inclusion may select a systematically different observation/severity population.',
 'The model does not establish biological phenotypes or treatment-response subgroups.',
 'If K4 has lowest BIC this only means best of tested K1-K4; it does not identify a globally optimal number of classes.'),
 sources=list(list(title='Estimation of Extended Mixed Models Using Latent Classes and Latent Processes: The R Package lcmm',authors='Proust-Lima C, Philipps V, Liquet B',year=2017,doi='10.18637/jss.v078.i02',url='https://www.jstatsoft.org/article/view/v078i02'),
 list(title='Official hlme documentation',url='https://cecileproust-lima.github.io/lcmm/reference/hlme.html'),
 list(title='Official gridsearch documentation',url='https://cecileproust-lima.github.io/lcmm/reference/gridsearch.html'),
 list(title='Official summarytable/summaryplot criteria',url='https://cecileproust-lima.github.io/lcmm/reference/summaryplot.html'))),
 path=file.path(root,'latent_model_methods.json'),auto_unbox=TRUE,pretty=TRUE)
print(model_table)
print(do.call(rbind,quality_rows))
cat('AGGREGATE_REPORT_COMPLETE\n')
