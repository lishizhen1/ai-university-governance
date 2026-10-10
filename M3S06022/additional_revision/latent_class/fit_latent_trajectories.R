# Formal growth-mixture sensitivity. No outcomes are loaded or used.
# Only outputs in private_local_only contain subject posteriors or model parameters.
args <- commandArgs(trailingOnly=TRUE)
root <- gsub('\\\\','/',args[1]) # Preserve ASCII junction alias on Windows.
mode <- if(length(args)>=2) args[2] else 'pilot'
# Dependencies are in this task's portable R runtime library.
library(lcmm)
library(jsonlite)
private <- file.path(root,'private_local_only')
variant <- if(length(args)>=4) args[4] else 'observed'
stopifnot(variant %in% c('observed','complete'))
dat <- read.csv(file.path(private,paste0('mimic_',variant,'_long.csv')))
expected_n <- if(variant=='observed') 10810L else 6826L
stopifnot(identical(names(dat),c('id','time','sic')),length(unique(dat$id))==expected_n)
if(variant=='complete') {
  root <- file.path(root,'complete_case')
  private <- file.path(root,'private_local_only')
}
dir.create(file.path(private,'models'),recursive=TRUE,showWarnings=FALSE)
dat$id <- as.numeric(dat$id)
set.seed(6022)
logline <- function(...) {cat(format(Sys.time()),..., '\n');flush.console()}
logline('START',mode,variant,'subjects',length(unique(dat$id)),'observations',nrow(dat))
one_path <- file.path(private,'models','gaussian_RI_K1.rds')
if(file.exists(one_path)) {
  m1 <- readRDS(one_path)
  stopifnot(identical(as.numeric(m1$pred$obs),as.numeric(dat$sic)),
            identical(as.numeric(m1$pred$id),as.numeric(dat$id)))
} else {
  m1 <- hlme(fixed=sic~time,random=~1,subject='id',ng=1,data=dat,
             maxiter=1000,convB=1e-5,convL=1e-5,convG=1e-5,verbose=FALSE)
  saveRDS(m1,one_path)
}
logline('K1',m1$conv,m1$loglik,m1$niter)
stopifnot(m1$conv==1)

diagnostic <- function(m,k,phase,start,seed,elapsed) {
  data.frame(k=k,phase=phase,start=start,seed=seed,conv=m$conv,loglik=m$loglik,
             parameters=length(m$best),AIC=m$AIC,BIC=m$BIC,niter=m$niter,
             gconv_parameters=m$gconv[1],gconv_likelihood=m$gconv[2],
             gconv_derivatives=m$gconv[3],elapsed_seconds=elapsed)
}
all_diagnostics <- diagnostic(m1,1,'full',1,6022,NA_real_)
write.csv(all_diagnostics,file.path(root,'model_initialization_diagnostics.csv'),row.names=FALSE)
if(mode=='pilot') {
  for(k in 2:4) {
    set.seed(6022+k)
    tm <- proc.time()[3]
    fit <- hlme(fixed=sic~time,mixture=~time,random=~1,subject='id',ng=k,data=dat,
                 B=random(m1),maxiter=30,convB=1e-5,convL=1e-5,convG=1e-5,verbose=FALSE)
    elapsed <- proc.time()[3]-tm
    saveRDS(fit,file.path(private,'models',paste0('pilot_K',k,'.rds')))
    all_diagnostics <- rbind(all_diagnostics,diagnostic(fit,k,'pilot',1,6022+k,elapsed))
    write.csv(all_diagnostics,file.path(root,'pilot_diagnostics.csv'),row.names=FALSE)
    logline('PILOT',k,'conv',fit$conv,'LL',fit$loglik,'seconds',elapsed)
  }
  quit(save='no')
}

# Short departures and subsequent full optimisations are reported individually.
# The highest fully converged likelihood, not a mortality contrast, selects each K fit.
short_reps <- if(length(args)>=3) as.integer(args[3]) else 40L
short_iter <- 30L
full_candidates <- 6L
selected <- list(m1)
workers <- parallel::makeCluster(4L,outfile=file.path(root,'parallel_short_starts.log'))
parallel::clusterEvalQ(workers,library(lcmm))
parallel::clusterExport(workers,c('dat','m1'))
short_worker <- function(s,k) {
  seed <- 6022000L+k*1000L+s
  set.seed(seed)
  tm <- proc.time()[3]
  fit <- tryCatch(hlme(fixed=sic~time,mixture=~time,random=~1,subject='id',ng=k,data=dat,
       B=random(m1),maxiter=30L,convB=1e-5,convL=1e-5,convG=1e-5,verbose=FALSE),error=function(e)e)
  list(fit=fit,seed=seed,elapsed=proc.time()[3]-tm)
}
for(k in 2:4) {
  logline('PARALLEL_SHORT_BEGIN',k,'replicates',short_reps,'workers',4L)
  jobs <- parallel::parLapplyLB(workers,seq_len(short_reps),short_worker,k=k)
  short <- vector('list',short_reps)
  for(s in seq_len(short_reps)) {
    seed <- jobs[[s]]$seed
    fit <- jobs[[s]]$fit
    elapsed <- jobs[[s]]$elapsed
    if(inherits(fit,'error')) {
      logline('ERROR_SHORT',k,s,conditionMessage(fit))
      next
    }
    short[[s]] <- fit
    all_diagnostics <- rbind(all_diagnostics,diagnostic(fit,k,'short',s,seed,elapsed))
    write.csv(all_diagnostics,file.path(root,'model_initialization_diagnostics.csv'),row.names=FALSE)
    logline('SHORT',k,s,'conv',fit$conv,'LL',fit$loglik,'seconds',elapsed)
  }
  ll <- vapply(short,function(x) if(is.null(x)) -Inf else x$loglik,numeric(1))
  candidates <- head(order(ll,decreasing=TRUE),full_candidates)
  full <- vector('list',length(candidates))
  for(j in seq_along(candidates)) {
    s <- candidates[j]
    tm <- proc.time()[3]
    fit <- tryCatch(hlme(fixed=sic~time,mixture=~time,random=~1,subject='id',ng=k,data=dat,
                  B=short[[s]]$best,maxiter=1000,convB=1e-5,convL=1e-5,convG=1e-5,verbose=FALSE),error=function(e)e)
    elapsed <- proc.time()[3]-tm
    if(inherits(fit,'error')) {logline('ERROR_FULL',k,s,conditionMessage(fit));next}
    full[[j]] <- fit
    saveRDS(fit,file.path(private,'models',paste0('gaussian_RI_K',k,'_start',s,'.rds')))
    all_diagnostics <- rbind(all_diagnostics,diagnostic(fit,k,'full',s,6022000L+k*1000L+s,elapsed))
    write.csv(all_diagnostics,file.path(root,'model_initialization_diagnostics.csv'),row.names=FALSE)
    logline('FULL',k,s,'conv',fit$conv,'LL',fit$loglik,'seconds',elapsed)
  }
  eligible <- which(vapply(full,function(x) !is.null(x) && x$conv==1,logical(1)))
  if(length(eligible)==0) {logline('NO_FULL_CONVERGENCE',k);next}
  best <- eligible[which.max(vapply(full[eligible],function(x)x$loglik,numeric(1)))]
  selected[[k]] <- full[[best]]
  saveRDS(selected[[k]],file.path(private,'models',paste0('gaussian_RI_K',k,'.rds')))
  logline('SELECTED_K',k,'LL',selected[[k]]$loglik,'BIC',selected[[k]]$BIC)
}
parallel::stopCluster(workers)
saveRDS(selected,file.path(private,'models','all_selected_RI.rds'))
logline('ALL_FITS_COMPLETE')
