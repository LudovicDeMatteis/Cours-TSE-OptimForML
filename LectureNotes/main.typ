#import "Templates/classes/lecture-notes.typ": notes

#show: notes.with(
  title: "Optimisation for machine learning",
  authors: (
    "Ludovic De Matteis": (
      "affiliation": "LAAS-CNRS",
      "email": "ldematteis@laas.fr",
    ),
  ),
  school: "Toulouse School of Economics",
)

#outline(title: "Table of contents")
#pagebreak()

#include "1.Motivations.typ"
#include "2.Derivatives.typ"
#include "3.GradientDescent.typ"
#include "4.NewtonMethod.typ"
#include "5.StochasticOptimization.typ"
#include "6.NeuralNetworks.typ"
#include "7.Opening.typ"
#include "8.Conclusion.typ"
