# Independent reload and zero-iteration likelihood verification after fit completion.
args <- commandArgs(trailingOnly=TRUE)
root <- gsub('\\\\','/',args[1])
library(lcmm);library(jsonlite)
results <- list()
restart_results <- list()
source_parses <- lapply(list.files(root,pattern='\\.R$',full.names=TRUE),function(p){invisible(parse(file=p));data.frame(file=basename(p),parsed=TRUE,md5=unname(tools::md5sum(p)))})
for(variant in c('observed','complete')) {
 dat <- read.csv(file.path(root,'private_local_only',paste0('mimic_',variant,'_long.csv')))
 dat$id <- as.numeric(dat$id)
 out <- if(variant=='observed') root else file.path(root,'complete_case')
 mdir <- file.path(out,'private_local_only','models')
 all_saved <- readRDS(file.path(mdir,'all_selected_RI.rds'))
 diagnostics <- read.csv(file.path(out,'model_initialization_diagnostics.csv'))
 stopifnot(nrow(diagnostics)==139,setequal(diagnostics$k,1:4))
 for(j in which(diagnostics$phase=='full' & diagnostics$k>1)) {
  row <- diagnostics[j,]
  saved <- readRDS(file.path(mdir,paste0('gaussian_RI_K',row$k,'_start',row$start,'.rds')))
  stopifnot(saved$conv==row$conv,saved$conv==1,saved$ns==length(unique(dat$id)),
   abs(saved$loglik-row$loglik)<1e-7,all(saved$gconv<1e-5),
   identical(as.numeric(saved$pred$obs),as.numeric(dat$sic)),
   identical(as.numeric(saved$pred$id),as.numeric(dat$id)))
  restart_results[[length(restart_results)+1]] <- data.frame(variant=variant,k=row$k,start=row$start,
   conv=saved$conv,loglik=saved$loglik,diagnostic_loglik_abs_difference=abs(saved$loglik-row$loglik),
   observed_input_sequence_matches=TRUE)
 }
 for(k in 1:4) {
  m <- readRDS(file.path(mdir,paste0('gaussian_RI_K',k,'.rds')))
  stopifnot(m$ng==k,m$conv==1,all(m$gconv<1e-5),m$ns==length(unique(dat$id)),
    identical(as.numeric(m$pred$obs),as.numeric(dat$sic)),
    identical(as.numeric(m$pred$id),as.numeric(dat$id)),
    identical(m$best,all_saved[[k]]$best))
  checked <- if(k==1) hlme(fixed=sic~time,random=~1,subject='id',ng=1,data=dat,B=m$best,maxiter=0) else
    hlme(fixed=sic~time,mixture=~time,random=~1,subject='id',ng=k,data=dat,B=m$best,maxiter=0)
  ll_difference <- abs(checked$loglik-m$loglik)
  pp_difference <- max(abs(as.matrix(checked$pprob[,paste0('prob',seq_len(k)),drop=FALSE])-as.matrix(m$pprob[,paste0('prob',seq_len(k)),drop=FALSE])))
  stopifnot(ll_difference<1e-6,pp_difference<1e-8)
  results[[length(results)+1]] <- data.frame(variant=variant,k=k,conv=m$conv,n_subjects=m$ns,
    model_file_md5=unname(tools::md5sum(file.path(mdir,paste0('gaussian_RI_K',k,'.rds')))),
    max_gconv=max(m$gconv),zero_iteration_loglik_abs_difference=ll_difference,
    posterior_max_abs_difference=pp_difference,
    observed_input_sequence_matches=TRUE,all_selected_checkpoint_matches=TRUE)
 }
}
write.csv(do.call(rbind,results),file.path(root,'saved_model_validation.csv'),row.names=FALSE)
write.csv(do.call(rbind,restart_results),file.path(root,'full_restart_validation.csv'),row.names=FALSE)
write.csv(do.call(rbind,source_parses),file.path(root,'R_source_parse_validation.csv'),row.names=FALSE)
write_json(list(status='PASS',n_models=length(results),
 n_full_restarts_independently_reloaded=length(restart_results),all_R_sources_parse=TRUE,
 verification='Reload all eight saved models, check exact observed SIC/id sequence, conv=1 and all convergence criteria<1e-5, and identical parameters in individual/checkpoint saves; independently recompute each likelihood and posterior at the saved parameters with maxiter=0.',
 reference_execution_note='In the original reference analysis, both long fitting processes completed all model fits, saved the all_selected checkpoint and printed ALL_FITS_COMPLETE; subsequently they exited 1 because edits to the executing source file caused stale stream offsets at end-of-file. The saved numeric fits were unaffected. The current scripts parse successfully and this independent full reload/likelihood verification exits successfully. No model fit is claimed successful solely from the interrupted process exit status.'),
 file.path(root,'execution_validation.json'),auto_unbox=TRUE,pretty=TRUE)
print(do.call(rbind,results));cat('INDEPENDENT_SAVED_MODEL_VALIDATION_PASS\n')
