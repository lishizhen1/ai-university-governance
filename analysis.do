capture log close
clear all
set more off
version 17

cd "."
log using "analysis_log.log", replace text

*-------------------------------------------------------------------------
* 1. Data import and preparation
*-------------------------------------------------------------------------
import delimited "data.csv", clear varnames(1)

label variable univ_id     "University identifier"
label variable year        "Survey year"
label variable province    "Province code"
label variable univ_type   "University type (1=research-intensive, 2=comprehensive/teaching, 3=applied/vocational)"
label variable ai_adopt    "AI adoption index"
label variable strat_gov   "Strategic governance capacity index"
label variable org_learn   "Organizational learning capability index"
label variable dig_cap     "Digital capability index"
label variable univ_scale  "University scale (log enrollment)"
label variable fin_invest  "Financial investment intensity"
label variable prof_ratio  "Faculty qualification ratio"

label define utype 1 "Research-intensive" 2 "Comprehensive/Teaching" 3 "Applied/Vocational"
label values univ_type utype

xtset univ_id year

*-------------------------------------------------------------------------
* 2. Descriptive statistics
*-------------------------------------------------------------------------
summarize ai_adopt strat_gov org_learn dig_cap univ_scale fin_invest prof_ratio
tabstat ai_adopt strat_gov org_learn dig_cap, by(univ_type) stat(mean sd n)
tabstat ai_adopt strat_gov org_learn dig_cap, by(year) stat(mean sd n)

correlate ai_adopt strat_gov org_learn dig_cap univ_scale fin_invest prof_ratio
pwcorr ai_adopt strat_gov org_learn dig_cap univ_scale fin_invest prof_ratio, sig star(0.05)

*-------------------------------------------------------------------------
* 3. Baseline regressions
*-------------------------------------------------------------------------
* Model 1: pooled OLS, no controls
regress strat_gov ai_adopt, vce(cluster univ_id)
estimates store m1

* Model 2: pooled OLS with controls and university-type fixed effects
regress strat_gov ai_adopt univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store m2
estat vif

* Model 3: pooled OLS with controls, type and year fixed effects
regress strat_gov ai_adopt univ_scale fin_invest prof_ratio i.univ_type i.year, vce(cluster univ_id)
estimates store m3

estimates table m1 m2 m3, star(0.1 0.05 0.01) stats(N r2)

*-------------------------------------------------------------------------
* 4. Panel fixed-effects and random-effects models
*-------------------------------------------------------------------------
xtreg strat_gov ai_adopt univ_scale fin_invest prof_ratio, fe vce(cluster univ_id)
estimates store fe1

xtreg strat_gov ai_adopt univ_scale fin_invest prof_ratio, re vce(cluster univ_id)
estimates store re1

quietly xtreg strat_gov ai_adopt univ_scale fin_invest prof_ratio, fe
estimates store fe_h
quietly xtreg strat_gov ai_adopt univ_scale fin_invest prof_ratio, re
estimates store re_h
hausman fe_h re_h, sigmamore

xtreg strat_gov ai_adopt univ_scale fin_invest prof_ratio i.year, fe vce(cluster univ_id)
estimates store fe2

*-------------------------------------------------------------------------
* 5. Mediation analysis: organizational learning and digital capability
*-------------------------------------------------------------------------
* Path a1: AI adoption -> organizational learning
regress org_learn ai_adopt univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store path_a1
local a1  = _b[ai_adopt]
local sea1 = _se[ai_adopt]

* Path a2: AI adoption -> digital capability
regress dig_cap ai_adopt univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store path_a2
local a2  = _b[ai_adopt]
local sea2 = _se[ai_adopt]

* Total effect (path c)
regress strat_gov ai_adopt univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store path_c
local c = _b[ai_adopt]

* Full model with mediators (path b and direct effect c')
regress strat_gov ai_adopt org_learn dig_cap univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store path_b
local b1  = _b[org_learn]
local seb1 = _se[org_learn]
local b2  = _b[dig_cap]
local seb2 = _se[dig_cap]
local cprime = _b[ai_adopt]

* Sobel test for the two indirect paths
local ind1 = `a1' * `b1'
local ind2 = `a2' * `b2'
local se_ind1 = sqrt((`b1'^2)*(`sea1'^2) + (`a1'^2)*(`seb1'^2))
local se_ind2 = sqrt((`b2'^2)*(`sea2'^2) + (`a2'^2)*(`seb2'^2))
local z1 = `ind1' / `se_ind1'
local z2 = `ind2' / `se_ind2'

display "----------------------------------------------------"
display "Indirect effect via organizational learning = " `ind1' "   Sobel z = " `z1'
display "Indirect effect via digital capability       = " `ind2' "   Sobel z = " `z2'
display "Total effect (c)      = " `c'
display "Direct effect (c')    = " `cprime'
display "Total indirect effect = " `ind1' + `ind2'
display "Proportion mediated    = " (`ind1' + `ind2') / `c'
display "----------------------------------------------------"

* Seemingly unrelated regression to obtain a joint covariance matrix
* for a delta-method test of the indirect effects
sureg (org_learn ai_adopt univ_scale fin_invest prof_ratio i.univ_type) ///
      (dig_cap ai_adopt univ_scale fin_invest prof_ratio i.univ_type) ///
      (strat_gov ai_adopt org_learn dig_cap univ_scale fin_invest prof_ratio i.univ_type)
nlcom (indirect_orglearn: _b[org_learn:ai_adopt]*_b[strat_gov:org_learn]) ///
      (indirect_digcap: _b[dig_cap:ai_adopt]*_b[strat_gov:dig_cap])

*-------------------------------------------------------------------------
* 6. Moderation analysis
*-------------------------------------------------------------------------
quietly summarize ai_adopt
generate ai_c = ai_adopt - r(mean)
quietly summarize org_learn
generate ol_c = org_learn - r(mean)
quietly summarize dig_cap
generate dc_c = dig_cap - r(mean)

generate ai_ol = ai_c * ol_c
generate ai_dc = ai_c * dc_c

regress strat_gov ai_c ol_c ai_ol univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store mod_orglearn

regress strat_gov ai_c dc_c ai_dc univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store mod_digcap

* Simple slopes of AI adoption at representative levels of digital capability
quietly summarize dig_cap
local sd_dc = r(sd)
lincom ai_c + `sd_dc' * ai_dc
lincom ai_c - `sd_dc' * ai_dc

*-------------------------------------------------------------------------
* 7. Heterogeneity analysis by university type
*-------------------------------------------------------------------------
forvalues t = 1/3 {
    regress strat_gov ai_adopt univ_scale fin_invest prof_ratio if univ_type == `t', vce(cluster univ_id)
    estimates store sub_type`t'
}
estimates table sub_type1 sub_type2 sub_type3, star(0.1 0.05 0.01) stats(N r2)

*-------------------------------------------------------------------------
* 8. Robustness: lagged explanatory variable
*-------------------------------------------------------------------------
xtset univ_id year
generate ai_adopt_lag = L.ai_adopt
regress strat_gov ai_adopt_lag univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store lag_model

*-------------------------------------------------------------------------
* 9. Robustness: alternative functional form
*-------------------------------------------------------------------------
generate ai_adopt_sq = ai_adopt^2
regress strat_gov ai_adopt ai_adopt_sq univ_scale fin_invest prof_ratio i.univ_type, vce(cluster univ_id)
estimates store quad_model

*-------------------------------------------------------------------------
* 10. Figures
*-------------------------------------------------------------------------
preserve
collapse (mean) ai_adopt strat_gov org_learn dig_cap, by(year)
twoway (line ai_adopt year, lcolor(blue) lwidth(medthick)) ///
       (line strat_gov year, lcolor(red) lwidth(medthick)) ///
       (line org_learn year, lcolor(green) lwidth(medthick)) ///
       (line dig_cap year, lcolor(purple) lwidth(medthick)), ///
       legend(order(1 "AI adoption" 2 "Strategic governance" ///
       3 "Organizational learning" 4 "Digital capability") pos(6) rows(2)) ///
       ytitle("Mean score") xtitle("Year") ///
       title("Trends in core constructs, 2020-2025")
graph export "fig_trends.png", replace width(1600)
restore

graph box ai_adopt, over(univ_type) ///
    ytitle("AI adoption index") title("AI adoption by university type")
graph export "fig_box_type.png", replace width(1600)

twoway (scatter strat_gov ai_adopt, msize(vsmall) mcolor(%40)) ///
       (lfit strat_gov ai_adopt, lcolor(black) lwidth(medthick)), ///
       ytitle("Strategic governance index") xtitle("AI adoption index") ///
       title("AI adoption and strategic governance") legend(off)
graph export "fig_scatter.png", replace width(1600)

*-------------------------------------------------------------------------
* 11. Summary tables to file
*-------------------------------------------------------------------------
estimates table m2 fe1 re1 lag_model, star(0.1 0.05 0.01) stats(N r2) ///
    title("Main and robustness estimates")

log close


