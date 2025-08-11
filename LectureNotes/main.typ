#import "include/imports.typ": *

#let title = "Optimisation for Machine Learning"
#let authors = "Ludovic De Matteis"
#let affiliations = "LAAS-CNRS"
#let emails = (link("mailto:ldematteis@laas.fr"))

#set page(
  paper: "a4",
  header: align(right)[Toulouse School of Economics],
  numbering: "1",
  number-align: center,
)
#set document(
  author: authors,
  title: title,
  date: auto,
)
#set par(justify: true)
#set text(
  font: "New Computer Modern",
  size: 11pt,
  spacing: 100%,
)
#set heading(numbering: "1.a")
#set math.equation(numbering: "(1)")
#set figure(
  numbering: "A",
  gap: 10pt,
)

#show heading: head => block(width: 100%, head + v(1%))
#show heading.where(level: 1): it => align(center)[
  #text(
    size: 14pt,
    weight: "semibold",
    smallcaps(it),
  )
]
#show figure.caption: capt => block(width: 75%, capt)

#align(
  center,
  text(size: 18pt)[#context document.title],
)

#let columns = ()
#let values = ()

#if type(authors) == str {
  grid(
    columns: 1fr,
    align(center)[
      #authors \
      #affiliations \
      #emails
    ]
  )
} else {
  for index in range(authors.len()) {
    let columns = columns.push(1fr)
    let values = values.push(
      align(center)[
        #authors.at(index) \
        #affiliations.at(index) \
        #emails.at(index)
      ],
    )
  }
  grid(
    columns: columns,
    ..values
  )
}


#outline(depth: 2, indent: 10%, title: "Table of contents")
#pagebreak()

#include "1.Motivations.typ"
#include "2.Derivatives.typ"

= Gradient Descent

== The algorithm

== Convergence Analysis

== Regularization

= Newton method

== Comparison the Gradient Descent

== Advantages

== Limits

= Stochastic Optimization

== Problem formulation

== Batch gradient descent

== Stochastic gradient descent

== Backpropagation

= Neural Networks

== Perceptron

== Multi-Layer Perceptron (MLP)

== Additional structures

= Opening

== Introduction to Large Language Models
