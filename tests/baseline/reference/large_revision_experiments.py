"""Large revision of the paper-facing numerical figures.

Theoretical = finite-dimensional deterministic equivalent.
Empirical = exact conditional risk for the fixed main-figure design.
Risk estimate = feasible plug-in risk under repeated outcomes for that design.
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from paper_requirements_aligned_core import (
    ARTIFACT_DIR, FIGURE_DIR, DEFAULT_CONFIG, COLORS, COMPONENT_COLORS,
    COMPONENT_DISPLAY_LABELS, SCENARIOS, configure_style,
    design_eigendecomposition, deterministic_equivalent_diagonal,
    deterministic_equivalent_identity_closed_form, exact_risk_components,
    external_base_weights, feasible_risk_geometry, mean_shift,
    ridge_augmented_path,
)
from spectral_quasi_score import (
    DEFAULT_THETA_BOUNDS, equation_41_estimates_many,
    spectral_quasi_score_from_feature_eigen,
)

CFG = DEFAULT_CONFIG
PART1 = ARTIFACT_DIR / "Part 1"
PART1.mkdir(parents=True, exist_ok=True)
configure_style()

COMPONENTS = ("Signal", "Residual noise", "Total")
COMPONENT_COLUMNS = {
    "Signal": ("exact_signal", "estimate_signal", "theoretical_signal"),
    "Residual noise": ("exact_residual", "estimate_residual", "theoretical_residual"),
    "Total": ("exact_total", "estimate_total", "theoretical_total"),
}


def _save(fig, stem, alias):
    for directory in (FIGURE_DIR, PART1):
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", pad_inches=.22, facecolor="white")
        fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=.22, facecolor="white")
    # Stable manuscript-facing alias.
    fig.savefig(FIGURE_DIR / alias, bbox_inches="tight", pad_inches=.22, facecolor="white")
    fig.savefig(PART1 / alias, bbox_inches="tight", pad_inches=.22, facecolor="white")
    plt.close(fig)


def _design_summary(frame, groups, columns):
    design = frame.groupby(groups + ["design"], as_index=False)[columns].mean()
    rows = []
    for keys, g in design.groupby(groups, sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(groups, keys)); row["designs"] = g.design.nunique()
        for col in columns:
            x = g[col].to_numpy(float)
            row[col] = x.mean(); row[f"{col}_two_se"] = 2*x.std(ddof=1)/np.sqrt(len(x))
        rows.append(row)
    return pd.DataFrame(rows)


def _nuisance_estimates(X0, Y0, eig, vec):
    mm_r, mm_s = equation_41_estimates_many(X0, Y0)
    sq_r, sq_s, _ = spectral_quasi_score_from_feature_eigen(
        X0, Y0, eig, vec, theta_bounds=DEFAULT_THETA_BOUNDS,
        pilot_r2=mm_r, pilot_sigma2=mm_s, grid_size=41,
    )
    return sq_r, sq_s


def _main_draws():
    """One fixed covariate design with repeated source outcomes."""
    cache = ARTIFACT_DIR / "revised_main_single_design_draws.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng = np.random.default_rng(CFG.seed + 9101)
    p, outcomes = 640, 200
    n0, n1 = round(p/CFG.phi0), round(p/CFG.phi1)
    X0=rng.normal(size=(n0,p)); target_noise=rng.normal(size=(n1,p))
    _,X0c,eig,vec=design_eigendecomposition(X0)
    beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes))
    eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
    Y0=X0@beta+eps; rhat,shat=_nuisance_estimates(X0,Y0,eig,vec)
    rows=[]
    for eta in CFG.etas:
        for scenario,(rho2,varrho2) in SCENARIOS.items():
            nu=mean_shift(p,n0,eta,rho2); X1=target_noise+nu
            gamma0=external_base_weights(n0,eta,varrho2)
            gp,delta=ridge_augmented_path(X0,X0c,eig,vec,X1.mean(0),gamma0,CFG.lambdas)
            exact=exact_risk_components(X0,gp,nu,CFG.r2,CFG.sigma2)
            sf,wn=feasible_risk_geometry(X1,delta,eig,vec,gp,CFG.lambdas)
            est_signal=rhat[:,None]*sf; est_residual=shat[:,None]*wn; est_total=est_signal+est_residual
            theory=deterministic_equivalent_diagonal(
                CFG.lambdas,phi0=p/n0,phi1=p/n1,eta=eta,rho2=rho2,varrho2=varrho2,
                r2=CFG.r2,sigma2=CFG.sigma2,
            )
            selected=np.argmin(est_total,axis=1)
            for draw in range(outcomes):
                for j,lam in enumerate(CFG.lambdas):
                    rows.append({"eta":eta,"scenario":scenario,"design":0,"draw":draw,"lambda":lam,
                      "exact_signal":n0**eta*exact[0][j],"exact_residual":n0**eta*exact[1][j],"exact_total":n0**eta*exact[2][j],
                      "estimate_signal":n0**eta*est_signal[draw,j],"estimate_residual":n0**eta*est_residual[draw,j],"estimate_total":n0**eta*est_total[draw,j],
                      "theoretical_signal":theory[0][j],"theoretical_residual":theory[1][j],"theoretical_total":theory[2][j],
                      "selected_lambda":CFG.lambdas[selected[draw]]})
    out=pd.DataFrame(rows); out.to_csv(cache,index=False,compression="gzip"); return out


def _single_design_summary(frame, groups, columns):
    rows=[]
    for keys,g in frame.groupby(groups,sort=False):
        if not isinstance(keys,tuple): keys=(keys,)
        row=dict(zip(groups,keys)); row["outcome_repetitions"]=g.draw.nunique()
        for col in columns:
            values=g[col].to_numpy(float)
            row[col]=values.mean()
            if col.startswith("estimate_"):
                # Conditional sampling SE of one plug-in estimate.  This is
                # deliberately not divided by sqrt(number of repetitions).
                row[f"{col}_conditional_se"]=values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def run_revised_main_figure():
    d=_main_draws()
    groups=["eta","scenario","lambda"]
    cols=[x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s=_single_design_summary(d,groups,cols)
    repetitions=int(s["outcome_repetitions"].iloc[0])
    for exact,_,_ in COMPONENT_COLUMNS.values():
        # Under the Gaussian predictive law, a squared centered prediction
        # error with variance R has variance 2 R^2.  This is the analytic SE
        # of the empirical mean of squared errors over the outcome repeats.
        s[f"{exact}_gaussian_se"]=np.sqrt(2.0/repetitions)*s[exact]
    tune=(d[["eta","scenario","design","draw","selected_lambda"]].drop_duplicates()
          .groupby(["eta","scenario"]).selected_lambda.apply(lambda x: float(np.exp(np.mean(np.log(x))))).reset_index())
    fig,axes=plt.subplots(3,3,figsize=(18.5,13.6),sharex=True,squeeze=False)
    point_idx=np.arange(1,len(CFG.lambdas),4)
    for r,eta in enumerate(CFG.etas):
        for c,scenario in enumerate(SCENARIOS):
            ax=axes[r,c]; q=s[(s.eta==eta)&(s.scenario==scenario)].sort_values("lambda")
            for comp in COMPONENTS:
                exact,estimate,theory=COMPONENT_COLUMNS[comp]; color=COMPONENT_COLORS[comp]
                ax.plot(q["lambda"],q[theory],color=color,ls="--",lw=2.0)
                ax.plot(q["lambda"],q[exact],color=color,lw=2.2)
                empirical_se=q[f"{exact}_gaussian_se"].to_numpy()
                empirical_mean=q[exact].to_numpy()
                ax.fill_between(q["lambda"].to_numpy(),np.maximum(empirical_mean-empirical_se,0.0),
                                empirical_mean+empirical_se,color=color,alpha=.12,lw=0)
                z=q.iloc[point_idx]
                ax.errorbar(z["lambda"],z[estimate],yerr=z[f"{estimate}_conditional_se"],color=color,
                            marker="D",mfc="white",mew=1.35,ls="none",ms=5.7,
                            capsize=4.0,capthick=1.45,elinewidth=1.45,zorder=7)
            selected=float(tune[(tune.eta==eta)&(tune.scenario==scenario)].selected_lambda.iloc[0])
            j=int(np.argmin(abs(q["lambda"].to_numpy()-selected)))
            ax.scatter(q.iloc[j]["lambda"],q.iloc[j].estimate_total,marker="*",s=285,color=COLORS["purple"],
                       edgecolor="white",linewidth=1.15,zorder=9)
            ax.set_xscale("log"); ax.margins(x=.035,y=.13)
            ax.grid(axis="x",alpha=.14); ax.grid(axis="y",alpha=.20)
            if r==0: ax.set_title(scenario,pad=10)
            if c==0: ax.set_ylabel(fr"$\eta={eta:g}$"+"\nScaled risk")
            if r==2: ax.set_xlabel(r"Penalty $\lambda$")
    component=[Line2D([0],[0],color=COMPONENT_COLORS[x],label=COMPONENT_DISPLAY_LABELS[x]) for x in COMPONENTS]
    types=[Line2D([0],[0],color=".2",ls="--",label="Theoretical"),Line2D([0],[0],color=".2",label="Empirical (±1 Gaussian SE)"),
           Line2D([0],[0],color=".2",marker="D",mfc="white",ls="none",label="Risk estimate (±1 conditional SE)"),
           Line2D([0],[0],color=COLORS["purple"],marker="*",ls="none",ms=16,label=r"Selected $\lambda$")]
    fig.legend(handles=component,loc="upper center",bbox_to_anchor=(.5,.995),ncol=3,
               frameon=False,handlelength=2.0,columnspacing=2.2)
    fig.legend(handles=types,loc="upper center",bbox_to_anchor=(.5,.963),ncol=4,
               frameon=False,handlelength=2.0,columnspacing=1.8)
    fig.subplots_adjust(left=.082,right=.986,bottom=.07,top=.895,hspace=.28,wspace=.23)
    _save(fig,"01_main_risk_theory_empirical_estimate","Main Figure.pdf")
    s.to_csv(ARTIFACT_DIR/"revised_main_risk_summary.csv",index=False)
    return {"draws":d,"summary":s}


def run_dimension_risk_comparison_diagnostic():
    d=_dimension_draws()
    groups=["p","eta","scenario"]
    cols=[x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s=_single_design_summary(d,groups,cols)
    fig,axes=plt.subplots(3,3,figsize=(19,14),sharex=True,squeeze=False)
    x=np.asarray(CFG.dimensions,dtype=float)
    for r,eta in enumerate(CFG.etas):
        for c,scenario in enumerate(SCENARIOS):
            ax=axes[r,c]
            q=(s[(s.eta==eta)&(s.scenario==scenario)]
               .set_index("p").reindex(CFG.dimensions).reset_index())
            for comp in COMPONENTS:
                exact,estimate,theory=COMPONENT_COLUMNS[comp]; color=COMPONENT_COLORS[comp]
                ax.plot(x,q[theory],color=color,ls="--",lw=2.0)
                ax.plot(x,q[exact],color=color,lw=2.2)
                ax.errorbar(x,q[estimate],yerr=q[f"{estimate}_conditional_se"],color=color,
                            marker="D",mfc="white",mew=1.35,ls="none",ms=5.7,
                            capsize=4.0,capthick=1.45,elinewidth=1.45,zorder=7)
            ax.set_xscale("log",base=2); ax.set_xticks(x,x.astype(int)); ax.margins(x=.08,y=.14)
            if r==0: ax.set_title(scenario,pad=10)
            if c==0: ax.set_ylabel(fr"$\eta={eta:g}$"+"\nScaled risk")
            if r==2: ax.set_xlabel(r"Dimension $p$")
    component=[Line2D([0],[0],color=COMPONENT_COLORS[x],label=COMPONENT_DISPLAY_LABELS[x]) for x in COMPONENTS]
    types=[Line2D([0],[0],color=".2",ls="--",label="Theoretical"),
           Line2D([0],[0],color=".2",label="Empirical"),
           Line2D([0],[0],color=".2",marker="D",mfc="white",ls="none",
                  label="Risk estimate ± 1 conditional SE")]
    fig.legend(handles=component+types,loc="upper center",bbox_to_anchor=(.5,.992),ncol=6,frameon=False)
    fig.text(.5,.018,r"Fixed penalty $\lambda=1$; one fixed covariate design and 200 outcome repetitions at each dimension.",
             ha="center",fontsize=11.5)
    fig.subplots_adjust(left=.085,right=.985,bottom=.08,top=.925,hspace=.28,wspace=.23)
    _save(fig,"02_dimension_risk_comparison","Uniform Error Figure.pdf")
    s.to_csv(ARTIFACT_DIR/"revised_dimension_risk_summary.csv",index=False)
    return {"draws":d,"summary":s}


def _dimension_draws():
    """Fixed-lambda risk comparison across dimensions, one design per p."""
    cache=ARTIFACT_DIR/"revised_dimension_single_design_draws.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng=np.random.default_rng(CFG.seed+9151)
    lam=np.array([1.0]); outcomes=200; rows=[]
    for p in CFG.dimensions:
        n0,n1=round(p/CFG.phi0),round(p/CFG.phi1)
        X0=rng.normal(size=(n0,p)); target_noise=rng.normal(size=(n1,p))
        _,X0c,eig,vec=design_eigendecomposition(X0)
        beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes))
        eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
        Y0=X0@beta+eps; rhat,shat=_nuisance_estimates(X0,Y0,eig,vec)
        for eta in CFG.etas:
            for scenario,(rho2,varrho2) in SCENARIOS.items():
                nu=mean_shift(p,n0,eta,rho2); X1=target_noise+nu
                gamma0=external_base_weights(n0,eta,varrho2)
                gp,delta=ridge_augmented_path(X0,X0c,eig,vec,X1.mean(0),gamma0,lam)
                exact=exact_risk_components(X0,gp,nu,CFG.r2,CFG.sigma2)
                sf,wn=feasible_risk_geometry(X1,delta,eig,vec,gp,lam)
                est_signal=rhat*sf[0]; est_residual=shat*wn[0]
                theory=deterministic_equivalent_diagonal(
                    lam,phi0=p/n0,phi1=p/n1,eta=eta,rho2=rho2,varrho2=varrho2,
                    r2=CFG.r2,sigma2=CFG.sigma2,
                )
                for draw in range(outcomes):
                    rows.append({"p":p,"eta":eta,"scenario":scenario,"design":0,"draw":draw,"lambda":1.0,
                      "exact_signal":n0**eta*exact[0][0],"exact_residual":n0**eta*exact[1][0],"exact_total":n0**eta*exact[2][0],
                      "estimate_signal":n0**eta*est_signal[draw],"estimate_residual":n0**eta*est_residual[draw],
                      "estimate_total":n0**eta*(est_signal[draw]+est_residual[draw]),
                      "theoretical_signal":theory[0][0],"theoretical_residual":theory[1][0],"theoretical_total":theory[2][0]})
    out=pd.DataFrame(rows); out.to_csv(cache,index=False,compression="gzip"); return out


def run_revised_error_figure():
    """Retained non-paper diagnostic for the uniform approximation errors."""
    de=pd.read_csv(ARTIFACT_DIR/"deterministic_equivalent_uniform_errors.csv")
    feasible=pd.read_csv(ARTIFACT_DIR/"uniform_risk_estimation_by_design.csv")
    etas=(0.,.5,1.); specs=[("Signal","Bias"),("Residual noise","Variance"),("Total","Total risk")]
    fig,axes=plt.subplots(3,4,figsize=(18.8,12.4),sharex="col",squeeze=False)
    colors={"Population shift only":COLORS["blue"],"Weight concentration only":COLORS["vermillion"],"Coupled":COLORS["green"]}
    for r,eta in enumerate(etas):
        for c,(comp,title) in enumerate(specs):
            ax=axes[r,c]; q=de[(de.eta==eta)&(de.component==comp)]
            for scenario,color in colors.items():
                z=q[q.scenario==scenario].groupby("p").uniform_absolute_error.agg(median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reindex(CFG.dimensions)
                x=np.asarray(CFG.dimensions); ax.plot(x,z["median"],color=color,marker="o",label=scenario)
                ax.fill_between(x,z.q25,z.q75,color=color,alpha=.13)
            ax.set_yscale("log"); ax.margins(x=.07,y=.15)
            ax.grid(axis="x",alpha=.12); ax.grid(axis="y",alpha=.20)
            if r==0: ax.set_title(title)
            if c==0: ax.set_ylabel(fr"$\eta={eta:g}$",rotation=0,labelpad=29,va="center")
        ax=axes[r,3]; q=feasible[(feasible.eta==eta)&(feasible.nuisance_input=="SQ plug-in")]
        z=q.groupby("p").scaled_uniform_risk_error.agg(median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reindex(CFG.dimensions)
        x=np.asarray(CFG.dimensions); ax.plot(x,z["median"],color=COLORS["purple"],marker="D")
        ax.fill_between(x,z.q25,z.q75,color=COLORS["purple"],alpha=.16)
        ax.set_yscale("log"); ax.margins(x=.07,y=.15)
        ax.grid(axis="x",alpha=.12); ax.grid(axis="y",alpha=.20)
        if r==0: ax.set_title("Feasible risk estimate")
    handles=[Line2D([0],[0],color=color,marker="o",label=name) for name,color in colors.items()]
    handles.append(Line2D([0],[0],color=COLORS["purple"],marker="D",label="Estimated nuisance"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.992),ncol=4,frameon=False)
    fig.supxlabel(r"Dimension $p$",y=.035)
    fig.subplots_adjust(left=.085,right=.985,bottom=.075,top=.91,hspace=.28,wspace=.20)
    _save(fig,"03_uniform_and_feasible_risk_errors","Uniform Error Figure.pdf")


def _aspect_draws():
    cache=ARTIFACT_DIR/"revised_aspect_single_design_rho_grid_draws.csv.gz"
    if cache.exists(): return pd.read_csv(cache)
    rng=np.random.default_rng(CFG.seed+9202); p=640; outcomes=200; lam=np.array([1.0]); phis=(.5,1.,2.,5.,10.)
    ratios=(.5,1.,5.,10.,20.); varrho2=1.0
    rows=[]
    def record(varied,phi,eta,ratio,design,X0,X1,nu,eig,vec,X0c,rhat,shat,gamma0):
        gp,delta=ridge_augmented_path(X0,X0c,eig,vec,X1.mean(0),gamma0,lam)
        exact=exact_risk_components(X0,gp,nu,CFG.r2,CFG.sigma2); sf,wn=feasible_risk_geometry(X1,delta,eig,vec,gp,lam)
        n0=len(X0); phi0=p/n0; phi1=p/len(X1)
        theory=deterministic_equivalent_identity_closed_form(lam,phi0=phi0,phi1=phi1,eta=eta,rho2=ratio,varrho2=varrho2,r2=CFG.r2,sigma2=CFG.sigma2)
        for draw in range(outcomes):
            es=rhat[draw]*sf[0]; ev=shat[draw]*wn[0]
            rows.append({"varied":varied,"phi":phi,"eta":eta,"rho2_over_varrho2":ratio,"design":design,"draw":draw,
              "exact_signal":n0**eta*exact[0][0],"exact_residual":n0**eta*exact[1][0],"exact_total":n0**eta*exact[2][0],
              "estimate_signal":n0**eta*es,"estimate_residual":n0**eta*ev,"estimate_total":n0**eta*(es+ev),
              "theoretical_signal":theory[0][0],"theoretical_residual":theory[1][0],"theoretical_total":theory[2][0]})
    # Source ratio: one fixed covariate design at each phi, reused across eta.
    for phi in phis:
        n0=round(p/phi); n1=round(p/CFG.phi1)
        X0=rng.normal(size=(n0,p)); _,X0c,eig,vec=design_eigendecomposition(X0)
        beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes)); eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
        rhat,shat=_nuisance_estimates(X0,X0@beta+eps,eig,vec); noise=rng.normal(size=(n1,p))
        for eta in (0.,1.):
            for ratio in ratios:
                nu=mean_shift(p,n0,eta,ratio); X1=noise+nu; gamma0=external_base_weights(n0,eta,varrho2)
                record("Source",phi,eta,ratio,0,X0,X1,nu,eig,vec,X0c,rhat,shat,gamma0)
    # Target ratio: one fixed source design reused over all target ratios.
    n0=round(p/CFG.phi0)
    X0=rng.normal(size=(n0,p)); _,X0c,eig,vec=design_eigendecomposition(X0)
    beta=rng.normal(scale=np.sqrt(CFG.r2/p),size=(p,outcomes)); eps=rng.normal(scale=np.sqrt(CFG.sigma2),size=(n0,outcomes))
    rhat,shat=_nuisance_estimates(X0,X0@beta+eps,eig,vec)
    for phi in phis:
        n1=round(p/phi); noise=rng.normal(size=(n1,p))
        for eta in (0.,1.):
            for ratio in ratios:
                nu=mean_shift(p,n0,eta,ratio); X1=noise+nu; gamma0=external_base_weights(n0,eta,varrho2)
                record("Target",phi,eta,ratio,0,X0,X1,nu,eig,vec,X0c,rhat,shat,gamma0)
    out=pd.DataFrame(rows); out.to_csv(cache,index=False,compression="gzip"); return out


def run_revised_aspect_figure():
    d=_aspect_draws(); groups=["varied","phi","eta","rho2_over_varrho2"]; cols=[x for trio in COMPONENT_COLUMNS.values() for x in trio]
    s=_single_design_summary(d,groups,cols); phi_curve=np.geomspace(.5,10,240)
    repetitions=int(s["outcome_repetitions"].iloc[0])
    for exact,_,_ in COMPONENT_COLUMNS.values():
        s[f"{exact}_gaussian_se"]=np.sqrt(2.0/repetitions)*s[exact]
    ratios=(.5,1.,5.,10.,20.)
    ratio_colors={.5:"#0072B2",1.:"#6C5CE7",5.:"#E31A1C",10.:"#E69F00",20.:"#087F23"}
    fig,axes=plt.subplots(3,4,figsize=(19.2,13.1),sharex="col",sharey="row",squeeze=False)
    panels=[("Source",0.),("Source",1.),("Target",0.),("Target",1.)]
    for c,(varied,eta) in enumerate(panels):
        phi0=phi_curve if varied=="Source" else np.full_like(phi_curve,CFG.phi0)
        phi1=phi_curve if varied=="Target" else np.full_like(phi_curve,CFG.phi1)
        for ratio in ratios:
            q=s[(s.varied==varied)&(s.eta==eta)&(s.rho2_over_varrho2==ratio)].sort_values("phi")
            curves={comp:[] for comp in COMPONENTS}
            for a,b in zip(phi0,phi1):
                th=deterministic_equivalent_identity_closed_form(
                    np.array([1.]),phi0=float(a),phi1=float(b),eta=eta,
                    rho2=ratio,varrho2=1.,r2=CFG.r2,sigma2=CFG.sigma2)
                for j,comp in enumerate(COMPONENTS): curves[comp].append(th[j][0])
            color=ratio_colors[ratio]
            for r,comp in enumerate(COMPONENTS):
                ax=axes[r,c]; exact,estimate,_=COMPONENT_COLUMNS[comp]
                ax.plot(phi_curve,curves[comp],color=color,ls="--",lw=1.75)
                ax.plot(q.phi,q[exact],color=color,lw=1.85)
                empirical_se=q[f"{exact}_gaussian_se"].to_numpy(); empirical_mean=q[exact].to_numpy()
                ax.fill_between(q.phi.to_numpy(),np.maximum(empirical_mean-empirical_se,0.0),
                                empirical_mean+empirical_se,color=color,alpha=.055,lw=0)
                ax.errorbar(q.phi,q[estimate],yerr=q[f"{estimate}_conditional_se"],color=color,
                            marker="D",mfc="white",mew=1.05,ls="none",ms=4.3,
                            capsize=2.6,capthick=1.05,elinewidth=1.05,zorder=7)
                ax.set_xscale("log"); ax.margins(x=.055,y=.15)
                ax.grid(axis="x",alpha=.13); ax.grid(axis="y",alpha=.20)
                if r==0: ax.set_title(("Source" if varied=="Source" else "Target")+fr" aspect ratio; $\eta={eta:g}$")
                if c==0: ax.set_ylabel(COMPONENT_DISPLAY_LABELS[comp])
                if r==2: ax.set_xlabel(r"Aspect ratio $\phi_0$" if varied=="Source" else r"Aspect ratio $\phi_1$")
    ratio_handles=[Line2D([0],[0],color=ratio_colors[x],lw=2.4,label=fr"$\rho^2/\varrho^2={x:g}$") for x in ratios]
    type_handles=[Line2D([0],[0],color=".2",ls="--",label="Theoretical"),
                  Line2D([0],[0],color=".2",label="Empirical ± 1 Gaussian SE"),
                  Line2D([0],[0],color=".2",marker="D",mfc="white",ls="none",label="Risk estimate ± 1 conditional SE")]
    fig.legend(handles=ratio_handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=5,frameon=False)
    fig.legend(handles=type_handles,loc="upper center",bbox_to_anchor=(.5,.958),ncol=3,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.07,top=.875,hspace=.25,wspace=.20)
    _save(fig,"02_aspect_ratio_risks","Aspect Ratio Figure.pdf")
    s.to_csv(ARTIFACT_DIR/"revised_aspect_ratio_summary.csv",index=False)
    return {"draws":d,"summary":s}


def run_revised_density_figure():
    d=pd.read_csv(ARTIFACT_DIR/"risk_tuning_calibration_draws.csv.gz")
    fig,axes=plt.subplots(2,2,figsize=(13.6,9.6),sharex=True,sharey=True)
    p_display=320
    if p_display not in set(d["p"].unique()):
        raise ValueError(f"Requested p={p_display} is absent from the calibration draws.")
    xs=np.linspace(-4.5,4.5,451)
    bins=np.linspace(-5.0,5.0,41)
    bin_width=float(bins[1]-bins[0])
    colors=(COLORS["blue"],COLORS["green"],COLORS["gold"],COLORS["vermillion"])
    for ax,eta,color in zip(axes.ravel(),CFG.calibration_etas,colors):
        values=d[(d.eta==eta)&(d.p==p_display)].studentized_error.to_numpy()
        # Divide by the full draw count and bin width, so the bars are on a
        # density scale without renormalizing after the finite plotting range.
        weights=np.full(values.shape,1.0/(len(values)*bin_width),dtype=float)
        ax.hist(values,bins=bins,weights=weights,color=color,alpha=.26,
                edgecolor="white",linewidth=.55,label=fr"Histogram ($p={p_display}$)")
        kde=stats.gaussian_kde(values,bw_method="scott")
        ax.plot(xs,kde(xs),color=color,lw=2.25,label="Kernel density")
        ax.plot(xs,stats.norm.pdf(xs),color=COLORS["black"],ls="--",lw=2.1,
                label=r"$N(0,1)$")
        ax.set_title(fr"$\eta={eta:g}$"); ax.margins(x=.025,y=.09)
        ax.grid(axis="x",alpha=.12); ax.grid(axis="y",alpha=.20)
    fig.supxlabel("Studentized prediction error",y=.055); fig.supylabel("Density",x=.035)
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.1,top=.91,hspace=.25,wspace=.20)
    _save(fig,"05_studentized_error_density","calibration density.pdf")


def run_all_dimension_density_figure():
    """Paper Figure 3: KDE comparison over all simulated dimensions."""
    d=pd.read_csv(ARTIFACT_DIR/"risk_tuning_calibration_draws.csv.gz")
    fig,axes=plt.subplots(2,2,figsize=(13.6,9.6),sharex=True,sharey=True)
    xs=np.linspace(-4.5,4.5,451)
    colors=(COLORS["blue"],COLORS["green"],COLORS["gold"],COLORS["vermillion"])
    for ax,eta in zip(axes.ravel(),CFG.calibration_etas):
        for p,color in zip(CFG.dimensions,colors):
            values=d[(d.eta==eta)&(d.p==p)].studentized_error.to_numpy()
            kde=stats.gaussian_kde(values,bw_method="scott")
            ax.plot(xs,kde(xs),color=color,lw=2.0,label=fr"$p={p}$")
        ax.plot(xs,stats.norm.pdf(xs),color=COLORS["black"],ls="--",lw=2.1,
                label=r"$N(0,1)$")
        ax.set_title(fr"$\eta={eta:g}$")
        ax.set_xlim(-4.5,4.5); ax.margins(y=.09)
        ax.grid(axis="x",alpha=.12); ax.grid(axis="y",alpha=.20)
    fig.supxlabel("Studentized prediction error",y=.055)
    fig.supylabel("Density",x=.035)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.99),ncol=5,
               frameon=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.1,top=.91,hspace=.25,wspace=.20)
    _save(fig,"03_distributional_calibration_all_dimensions",
          "distributional calibration all dimensions.pdf")
