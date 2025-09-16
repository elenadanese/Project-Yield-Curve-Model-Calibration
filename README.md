## Project Description

This group project, developed as part of my MSc in Computational Finance, focused on modeling and analyzing yield curves using the Nelson-Siegel and Svensson frameworks. We employed real bond yield data at short and medium maturities from three markets: Italy, the United States, and Japan. The objective was not only to fit the curves but also to critically compare the models from mathematical, computational, and economic perspectives.

From a technical standpoint, parameter estimation was performed through two numerical optimization methods: **Gradient Descent** and **Newton’s Method**. This allowed us to investigate how different algorithms behave in terms of convergence speed, sensitivity to step size and initial values, and robustness in practice. The comparison highlighted strengths and weaknesses of each optimization approach when applied to yield curve modeling.

Beyond the computational side, the project emphasized **economic interpretation**. By fitting Nelson-Siegel and Svensson curves to real-world data, we observed how bond markets across different geographies responded differently to global events, particularly the economic shocks caused by Covid-19. 

Key strengths of this project included the use of real financial data, the critical comparison of competing models, and the integration of mathematical, computational, and economic perspectives. The collaborative nature of the project also fostered discussion within the team, allowing us to connect theoretical models to practical implications in fixed income markets.

## Summary of Results

Overall, the analysis showed that **Newton’s Method** is more robust and accurate than **Gradient Descent**, especially when the starting point is poorly chosen or when the step size in GD is not properly calibrated. While Gradient Descent is faster in terms of execution time, it is highly sensitive to step size and often diverges, whereas Newton achieves more reliable convergence at a higher computational cost.  

From an economic perspective, the results confirmed that yield curves across different markets reacted differently to global shocks. The ECB and Fed adopted aggressive tightening policies, leading to steeper short-term yields, while the BoJ maintained a stable zero-rate policy, producing flatter curves. 

## Alternative Calibration Methods (Not Implemented)

In addition to Gradient Descent and Newton’s Method, two other algorithms are widely used for yield curve calibration:  

- **Levenberg–Marquardt (LM)**: Combines the stability of Gradient Descent with the fast convergence of Gauss-Newton by adaptively adjusting a damping parameter. This makes it particularly effective for non-linear least squares problems, though it still requires matrix inversion, which can be costly for large-scale models.  

- **Powell’s Method**: A derivative-free optimization algorithm that relies on successive linear searches along adaptive directions. It is especially useful when gradients are difficult to compute or the objective function is noisy, although it generally requires more function evaluations compared to gradient-based methods.  

While not implemented in this project, these methods represent promising alternatives for future extensions, offering different trade-offs between accuracy, robustness, and computational efficiency.

## Tools and Libraries

The project was implemented in **Python** using the following main libraries:

- **NumPy**: numerical computation and array handling  
- **Pandas**: data loading and preprocessing  
- **Matplotlib**: data visualization and yield curve plotting  
- **SciPy**: gradient approximation and linear algebra utilities.  
- **time** and **os**: performance measurement and file handling.  

The full report was developed as a group project during the MSc in Computational Finance. For privacy reasons, only the technical sections curated by me are included here.
