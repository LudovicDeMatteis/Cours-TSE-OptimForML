#import "include/imports.typ": *

= Gradient Descent
== Algorithm
The gradient descent algorithm is an iterative method to solve the minimization problem @min_problem when the function $f$ is differentiable. The algorithm starts from an initial point $x_0 in RR^n$ and iteratively updates the current point $x_k$ using the formula
$ x_(k+1) = x_k - alpha_k nabla f(x_k) $
where $alpha_k > 0$ is the step size at iteration $k$.
The algorithm stops when a stopping criterion is met, for instance when the maximum number of iterations is reached or when the norm of the gradient is below a certain threshold. The complete algorithm is summarized in algo @gradient-descent.

#grid(
  columns: (auto, auto),
  gutter: 5%,
  [
    #algorithm-figure("Gradient Descent", vstroke: .5pt + luma(200), {
      import algorithmic: *
      Procedure("Gradient Descent", ($x_0$, $f$), {
        Comment[Initialize the solution]
        Assign[$x$][$x_0$]
        While($x_0 "is not optimal"$, {
          Comment[Compute the gradient at the current solution]
          Assign[$g$][$nabla f(x)$]
          Comment[Choose a step size]
          Assign[$alpha$][*some method*]
          Comment[Update the solution]
          Assign[$x$][$x - alpha g$]
        })
      })
    }) <gradient-descent>
  ],
  align(horizon + center)[
    #image("Images/gradient_descent_1.jpg", width: 80%),
  ],
)

== Linesearch methods
A fundamental component of the gradient descent algorithm is the choice of the step size (or learning rate) $alpha_k$. Several methods exist to choose this parameter, the most common ones being:
- Fixed step size: The step size is set to a constant value $alpha_k = alpha$ for all iterations. This method is simple to implement but can lead to slow convergence or divergence if the step size is not well chosen.

== Limitations
The gradient descent method has several limitations:
- It can be slow to converge, especially for ill-conditionned systems for which the gradient can lead to slow progress towards the minimum.
- Gradient descent can get stuck in local minima or saddle points, especially for non-convex functions.


