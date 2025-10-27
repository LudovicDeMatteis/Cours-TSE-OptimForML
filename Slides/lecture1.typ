#import "Templates/slides/standard.typ": *
#import "@preview/polylux:0.4.0": *
#import "@preview/subpar:0.2.2"
#import "@preview/algorithmic:1.0.6"
#import "Templates/global/math.typ": *
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

#show: slides.with(
  title: "Optimization for deep learning",
  subtitle: "Toulouse School of Economics",
  authors: (
    (
      name: "Ludovic De Matteis",
      affiliation: "LAAS-CNRS",
      email: "ldematteis@laas.fr",
    ),
  ),
  footer: "Ludovic De Matteis - Optimization for deep learning",
)

#content_slide(
  title: "Course Overview",
  subtitle: "Teacher",
)[
  #align(center)[
    #text(
      size: 24pt,
      weight: "bold",
      "Ludovic De Matteïs",
    )\
    #link("mailto:ldematteis@laas.fr") - 0781390605
  ] \
  - PhD Student in robotics at LAAS-CNRS
    - Research interests in optimization and optimal control

  - Studies at Ecole Normale Superieure (ENS) Paris-Saclay
    - Master in Electrical Engineering
    - Master in Mathematics, Vision and Learning
    - Agrégation in Engineering - specialty in Computer Science
]

#content_slide(
  title: "Course Overview",
  subtitle: "Planning",
)[
  #align(center + horizon)[
    #table(
      columns: 4,
      [], [Day], [Time], [Subject],

      [1], [13/10/2025], [9:30 - 12:30], [Basic definitions, Gradient Descent and Newton's method],
      [2], [27/10/2025], [9:30 - 12:30], [Practical - Gradient Descent and Newton's method],
      [3], [03/11/2025], [9:30 - 12:30], [Neural networks and stockastic gradient descent],
      [4], [10/11/2025], [9:30 - 12:30], [Practical - Neural Networks and digit recognition],
      [5], [17/11/2025], [9:30 - 12:30], [Alternative Neural Structures],
      [6], [24/11/2025], [9:30 - 12:30], [Practical - Alternative Neural structure, Adversial networks],
    )
  ]
]

#content_slide(
  title: "Course Overview",
  subtitle: "Evaluation",
)[
  - No final exam
  - 3 pratical session
    - 1 report per session
    - 2 weeks delay to complete each report
    - 60% of the final grade
  - 1 MCQ at the end of the course
    - 40% of the final grade
]

#title_slide(
  title: "Lecture 1 -",
  subtitle: "On gradients and optimization algorithms",
)

#content_slide(
  title: "Summary",
)[
  #align(horizon)[
    + Motivation in Machine Learning
      - What is machine learning?
      - What is optimization?
      - Classical problems in machine learning
    + Derivatives and gradients
      - Reminder on derivatives
      - Optimality conditions
    + Gradient descent algorithm
      - Algorithm
      - Limitations
    + Newton's method
      - Comparison with gradient descent
      - Algorithm
      - Limitations
  ]
]

#title_slide(
  title: "Motivation in Machine Learning",
)

#content_slide(title: "Motivation in Machine Learning", subtitle: "What is machine learning?")[
  - Subfield of artificial intelligence that focuses on the development of algorithms that enable computers to perform specific tasks without explicit instructions.
  - Systems learn from and make predictions or decisions based on data.
  #grid(
    columns: (auto, auto),
    gutter: 5%,
    [
      - The primary goal of machine learning is to enable computers to improve their performance on a task over time as they are exposed to more data.
      - Widely used in various applications, including image and speech recognition, natural language processing, recommendation systems, robotics, finances...
    ],
    image("Images/Machine-Learning-ia.png", width: 65%),
  )
]

#content_slide(
  title: "Motivation in Machine Learning",
  subtitle: "What is machine learning?",
)[
  - Three main classes of machine learning:
    - *Supervised learning*: The model is trained on a labeled dataset, where the input data is paired with the correct output. The goal is to learn a mapping from inputs to outputs.
    - *Unsupervised learning*: The model is trained on an unlabeled dataset, where the input data does not have corresponding output labels. The goal is to discover patterns or structures in the data.
    #grid(
      columns: (auto, auto),
      gutter: 5%,
      [
        - *Reinforcement learning*: The model learns to make decisions by interacting with an environment. It receives feedback in the form of rewards based on its actions and aims to maximize cumulative rewards over time.
      ],
      image("Images/Machine-Learning-classes.jpg", width: 80%),
    )
]

#content_slide(
  title: "Motivation in Machine Learning",
  subtitle: "What is optimization?",
)[
  - Optimization is the process of finding the best solution to a problem from a set of possible solutions, often by minimizing or maximizing a specific objective function.
  - It can be written (in the case a minimization as)
  $
    inf_(x in cal(X)) f(x)
  $
  where $f: cal(X) arrow.r RR$ is the *objective function*, $x$ are the *decision variables* and $cal(X)$ is the *set of feasible points*.
  - In machine learning, optimization is used to find the best parameters for a model to minimize a loss function that measures the difference between the model's predictions and the actual data.
  - The best decision variable is called the *optimal solution* and is denoted $x^*$.
]

#content_slide(
  title: "Motivation in Machine Learning",
  subtitle: "What is optimization?",
)[
  - We define the set of *global minimizers* of the function $f$ as
  $ arg min_(x in RR^n) f(x) =^"def" {x_0 in RR^n | forall x in RR^n, f(x_0) <= f(x)} $
  - The global minimizer of a function does not necessarily exists and if it does i can be non unique.
  #subpar.grid(
    image("Images/function_convexe.png"),
    image("Images/250px-Nonquasiconvex_function.png"),
    image("Images/X_Cubed.svg.png"),

    columns: (30%, 21%, 18%),
  )
]

#content_slide(
  title: "Motivation in Machine Learning",
  subtitle: "What is optimization?",
)[
  - We will also define the notion of *local minimizer* as follows. The point $x^*$ is a local minimizer of the function $f$ if there exists a radius $r > 0$ such that for all $x in RR^n$ such that $||x - x^*|| <= r$, we have $f(x^*) <= f(x)$.
  - Note that a local minimizer is not necessarily a global minimizer (but a global minimizer is necessarly a local minimizer).
  #subpar.grid(
    image("Images/local_min.png"), image("Images/local_min2.png"),

    columns: (39%, 30%),
  )
]

#content_slide(
  title: "Motivation in Machine Learning",
  subtitle: "Classical problems in machine learning",
)[
  #align(center)[== Regression]
  - The problem of regression consists in finding a function that best fits a set of data points.
  - We will start by considering a linear regression problem, in which
  $ f(x) = 1 / 2 sum_(i-1)^N (y_i - <x, a_i>)^2 = 1 / 2 || A x - y ||^2 $
  is the least square quadratic risk function.
  #subpar.grid(
    image("Images/linear_regression.png", height: 2in, fit: "contain"), image("Images/quad_regression.png"),

    columns: (39%, 30%),
  )
]

#content_slide(
  title: "Motivation in Machine Learning",
  subtitle: "Classical problems in machine learning",
)[
  #align(center)[== Classification]
  - For binary classification, the data points $y_i$ are *labelled* with a value $1$ or $-1$, defining a class.
  - In this problem, we aim the minimize the function
  $ f(x) = sum_(i-1)^n l(-y_i<x, a_i>) = L(-"diag"(y)A x) $
  where $l$ is the 0-1 loss function $1_(RR^+)$ or a smooth approximation of it, giving a value of 1 if the signs of $y_i$ and $<a_i, x>$ are opposed and zero otherwise.
  #align(center)[
    #box(image("Images/classification.png", height: 2.5in, fit: "cover"), clip: true, inset: (
      bottom: -1cm,
      top: -1cm,
      left: -2.5cm,
      right: -2.6cm,
    ))
  ]
]

#title_slide(
  title: "Derivatives and gradients",
)

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "Reminder on derivatives",
)[
  - A function $f: RR -> RR$ is said to be differentiable at a point $x_0 in RR$ if the following limit exists
  $ lim_(h->0) (f(x_0 + h) - f(x_0)) / h $
  The value of this limit when it exists is called the derivative of $f$ at $x_0$ and is denoted $f'(x_0)$ or $(d f) / (d x) (x_0)$.
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "Reminder on derivatives",
)[
  #align(center)[_Examples_]
  - Quadratic function: $f(x) = 3 x^2 - x$
  #v(5%)
  - Exponential function: $f(x) = e^(-2x)$
  _Note_: we know that the limit when $h->0$ of $((e^(h) - 1)) / h$ is $1$ (it can be shown using the Taylor expansion of the exponential function).
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "On gradients and Jacobians",
)[
  #align(center)[== Gradient]
  #v(5%)
  - Extend derivative definition to a function of multiple variables $f: RR^d -> RR$
  - The function $f$ is differentiable at a point $x_0 in RR^d$ if there exists a vector $g in RR^d$ such that
  $ f(x_0 + h) = f(x_0) + g^T h + o(||h||) $
  where $o(||h||)$ is a function such that $o(||h||) / (||h||) -> 0$ when $||h|| -> 0$.
  - The vector $g$ is called the gradient of $f$ at point $x_0$ and is denoted $nabla f(x_0)$ or $(d f) / (d x) (x_0)$.
  - The gradient vector is related to the derivatives of the function by
  $ nabla f(x) = ( (d f) / (d x_1) (x), (d f) / (d x_2) (x), ..., (d f) / (d x_d) (x) )^T $
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "On gradients and Jacobians",
)[
  #align(center)[== Jacobian]
  #v(5%)
  - We define the *Jacobian* of a function $f: RR^d -> RR^p$ as the matrix $J in RR^(p times d)$ writen as
  $
    J & = mat(
          (d f_1) / (d x_1), (d f_1) / (d x_2), ..., (d f_1) / (d x_d);
          (d f_2) / (d x_1), (d f_2) / (d x_2), ..., (d f_2) / (d x_d);
          dots.v, dots.v, dots.down, dots.v;
          (d f_p) / (d x_1), (d f_p) / (d x_2), ..., (d f_p) / (d x_d);
        ) \
      & = mat(
          nabla f_1(x)^T;
          nabla f_2(x)^T;
          dots.v;
          nabla f_p (x)^T
        )
  $
  where $f_i$ is the $i$-th component of the vector-valued function $f$.
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "The chain rule",
)[
  - The chain rule is a fundamental theorem in calculus that describes how to compute the derivative of a composite function.
  - If we have two functions $g: RR^m -> RR^p$ and $f: RR^d -> RR^m$, the composite function $h: RR^d -> RR^p$ is defined as $h(x) = g(f(x))$.
  - The chain rule states that if $f$ is differentiable at a point $x in RR^d$ and $g$ is differentiable at the point $f(x) in RR^m$, then the composite function $h$ is differentiable at point $x$, and its derivative is given by
  $ J_h(x) = J_g(f(x)) J_f(x) $
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "The chain rule",
)[
  #align(center)[_Example_]
  - Let $f: RR^2 -> RR^2$ and $g: RR^2 -> RR$ be defined as
  $ f(x) = mat(x_1^2, x_2^3)^T wide g(y) = y_1 + y_2 $
  - The Jacobian of $f$ and the gradient of $g$ are
  $ J_f(x) = mat(2 x_1, 0; 0, 3 x_2^2) wide nabla g(y) = (1, 1)^T $
  - The composite function $h: RR^2 -> RR$ is defined as
  $ h(x) = g(f(x)) = x_1^2 + x_2^3 $
  - The gradient of $h$ at point $x$ is given by
  $ nabla h(x) = J_f(x)^T nabla g(f(x)) = (2 x_1, 3 x_2^2)^T $
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "Optimality conditions",
)[
  - Let's consider the unconstrained optimization problem
  $ min_(x in RR^n) f(x) $
  - If $f$ is differentiable and $x^*$ is a local minimizer of $f$, then the following condition holds
  $ nabla f(x^*) = 0 $
  - Let's prove this result
]

#content_slide(
  title: "Derivatives and gradients",
  subtitle: "Optimality conditions",
)[
  #proof[
    By definition of a local minimizer, there exists a radius $r > 0$ such that for all $x in RR^d$ such that $||x - x^*|| <= r$, we have $f(x^*) <= f(x)$.
    This means that for $h in RR^d$ such that $||h|| <= r$, we have
    $ f(x^*) <= f(x^* + h) = f(x^*) + nabla f(x^*)^T h + o(||h||) $
    Simplifying by $f(x^*)$ and dividing by $||h||$ (which is non zero as $h$ is non zero), we obtain,
    $ nabla f(x^*)^T overline(h) >= 0 $
    with $overline(h) = h / (||h||)$. \
    We can apply the same reasoning with $-h$ instead of $h$ and obtain $nabla f(x^*)^T (-overline(h)) >= 0$, which gives $nabla f(x^*)^T overline(h) <= 0$.\
    Eventually, we have $nabla f(x^*)^T overline(h) = 0$ for all $overline(h)$ such that $||overline(h)|| = 1$, and thus $nabla f(x^*) = 0$
    #align(right)[$qed$]
  ]
]

#title_slide(
  title: "Gradient descent algorithm",
)

#content_slide(
  title: "Gradient descent algorithm",
  subtitle: "Algorithm",
)[
  - The gradient descent algorithm is an iterative optimization algorithm used to find the minimum of a differentiable function.
  - The basic idea behind gradient descent is to iteratively update the current solution in the direction of the negative gradient of the objective function, which points towards the steepest descent.
  - The update rule for gradient descent is given by
  $ x_(k+1) = x_k - alpha_k nabla f(x_k) $
  where $x_k$ is the current solution, $alpha_k$ is the step size (or learning rate), and $nabla f(x_k)$ is the gradient of the objective function at point $x_k$.
  - The choice of step size $alpha_k$ is crucial for the convergence and performance of the algorithm. It can be fixed or determined using techniques such as line search or adaptive methods.
]

#content_slide(
  title: "Gradient descent algorithm",
  subtitle: "Algorithm",
)[
  - The pseudocode for the gradient descent algorithm is as follows:
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
      })
    ],
    align(horizon + center)[
      #image("Images/gradient_descent_1.jpg", width: 60%),
    ],
  )
]

#content_slide(
  title: "Gradient descent algorithm",
  subtitle: "Limitations",
)[
  - Gradient descent can be sensitive to the choice of step size. A step size that is too large can lead to divergence, while a step size that is too small can result in slow convergence.
  - The algorithm can get stuck in local minima or saddle points, especially in non-convex optimization problems.
  - Gradient descent may require many iterations to converge, particularly for ill-conditioned problems where the objective function has steep and flat regions.
  - Gradient descent only uses first-order information (the gradient) and does not take into account the curvature of the objective function, which can lead to inefficient updates.
]

#title_slide(
  title: "Newton's method",
)

#content_slide(
  title: "Newton's method",
  subtitle: "Comparison with gradient descent",
)[
  - Newton's method is an iterative optimization algorithm used to find the minimum of a *twice-differentiable* function.
  - Unlike gradient descent, which only uses first-order information (the gradient), Newton's method also incorporates second-order information (the Hessian matrix) to make more informed updates.
  - The update rule for Newton's method is given by
  $ x_(k+1) = x_k - H_f(x_k)^(-1) nabla f(x_k) $
  where $H_f(x_k)$ is the Hessian matrix of the objective function at point $x_k$.
  - By using the Hessian, Newton's method can adapt the step size and direction based on the local curvature of the objective function, potentially leading to faster convergence compared to gradient descent.
]

#content_slide(
  title: "Newton's method",
  subtitle: "Finding zeroes of a function",
)[
  - The original Newton's method is used to find the roots (zeroes) of a function $g: RR^n -> RR^n$.
  - The update rule for finding the roots of $g$ is given by
  $ x_(k+1) = x_k - J_g(x_k)^(-1) g(x_k) $
  #align(center)[#image("Images/newtons-method-calculus.png", width: 40%)]
  - This can be applied to get the zero of the gradient of a function to minimize and yields the previous iterate
]

#content_slide(
  title: "Newton's method",
  subtitle: "Algorithm",
)[
  - The pseudocode for Newton's method is as follows:
  #grid(
    columns: (auto, auto),
    gutter: 5%,
    [
      #algorithm-figure("Binary Search", vstroke: .5pt + luma(200), {
        import algorithmic: *
        Procedure("Newton's Method", ($x_0$, $f$), {
          Comment[Initialize the solution]
          Assign[$x$][$x_0$]
          While($x_0 "is not optimal"$, {
            Comment[Compute the gradient and Hessian at the current solution]
            Assign[$g$][$nabla f(x)$]
            Assign[$H$][$H_f(x)$]
            Comment[Compute the step size]
            Assign[$alpha$][Some method]
            Comment[Update the solution]
            Assign[$x$][$x - alpha H^(-1) g$]
          })
        })
      })
    ],
  )
]

#content_slide(
  title: "Newton's method",
  subtitle: "Limitations",
)[
  - Computing the Hessian matrix and its inverse can be computationally expensive, especially for high-dimensional problems, which can limit the scalability of Newton's method.
  - The analytical computation of the Hessian may not be feasible for complex functions, requiring numerical approximations that can introduce errors
  - Newton's method can be sensitive to the initial guess. A poor initial guess may lead to slow convergence or divergence.
  - In practice, quasi-Newton methods (e.g., BFGS) are often used as they approximate the Hessian and can provide a good trade-off between convergence speed and computational cost.
]


