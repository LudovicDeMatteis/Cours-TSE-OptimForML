#import "include/imports.typ": *

= Motivations in Machine Learning
Optimization describe a mathematical theory focussing on the problem of finding the minimum (or equivalently the maximum) of a function. This problems occurs in almost every domain, from investment portfolios built to maximize the return to minimizing the error in a weather forecast model.
This introductory course aims at teaching the basics of optimization and how it is applied to solve machine learning problem.
== Unconstraint Optimization
In unconstraint optimization, we consider problems of the form
$ inf_(x in RR^n) f(x) $ <inf_problem>
we define the set of *global minimizers* of the function $f$ as
$ arg min_(x in RR^n) f(x) =^"def" {x_0 in RR^n | forall x in RR^n, f(x_0) <= f(x)} $
The global minimizer of a function does not necessarily exists and if it does, it can be non unique.
#subpar.grid(
  figure(image("Images/function_convexe.png"), caption: [Unique Minimizer]),
  <unique_min>,
  figure(image("Images/250px-Nonquasiconvex_function.png"), caption: [Multiple Minimizer]),

  <multiple_min>, figure(image("Images/X_Cubed.svg.png"), caption: [Zero Minimizer]), <zero_min>,
  columns: (1.5fr, 1fr, 0.7fr),
  caption: [Different functions shapes and corresponding number of minimizers],
  label: <minimizers>,
)
@minimizers shows different function shapes corresponding to different number of minimizers.
In general, when a global minimizer exists - and this will be our focus for the rest of the course - we denote the problem @inf_problem as
$ min_(x in RR^n) f(x) $ <min_problem>

We will also define the notion of *local minimizer* as follows. The point $x^*$ is a local minimizer of the function $f$ if there exists a radius $r > 0$ such that for all $x in RR^n$ such that $||x - x^*|| <= r$, we have $f(x^*) <= f(x)$.
Note that a local minimizer is not necessarily a global minimizer (but a global minimizer is necessarly a local minimizer).

== Regression
We will start by considering a linear regression problem, in which
$ f(x) = 1 / 2 sum_(i-1)^N (y_i - <x, a_i>)^2 = 1 / 2 || A x - y ||^2 $
is the least square quadratic risk function.
#figure(
  box(
    image("Images/linear_regression.png", height: 2.5in, fit: "contain"),
    clip: true,
  ),
  caption: [Linear regression problem. \
    The linear function $l(x) = A x$ in red aims to fit as well as possible the given data points],
) <linear_regre>

An illustration of the linear regression problem is shown in @linear_regre.

The regression problem can be extended to different function, by considering for instance a quadratic function
$ f(x) = 1 / 2 ||(x^T A x + B x) - y||^2 $
or an exponential function
$ f(x) = 1 / 2 || e^(A x) - y ||^2 $

== Classification
For binary classification, the data points $y_i$ are *labelled* with a value $1$ or $-1$, defining a class.
In this problem, we aim the minimize the function
$ f(x) = sum_(i-1)^n l(-y_i<x, a_i>) = L(-"diag"(y)A x) $
where $l$ is the 0-1 loss function $1_(RR^+)$ or a smooth approximation of it, giving a value of 1 if the signs of $y_i$ and $<a_i, x>$ are opposed and zero otherwise.
#figure(
  box(image("Images/classification.png", height: 3in, fit: "cover"), clip: true, inset: (
    bottom: -1cm,
    top: -1cm,
    left: -2.5cm,
    right: -2.6cm,
  )),
  caption: [Binary Classification problem. \
    The function in black aims to separate as well as possible the different classes],
) <binary_class>
As for regression, the classification problem can use a different separating function.
@binary_class presents a binary classification problem with an arbitrary separating function.


