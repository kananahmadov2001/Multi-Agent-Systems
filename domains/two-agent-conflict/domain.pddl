(define (domain grid-conflict)
  (:requirements :strips :typing)
  (:types agent cell)
  (:predicates
    (at ?a - agent ?c - cell)
    (free ?c - cell)
    (adj ?from - cell ?to - cell))
  (:action move
    :parameters (?a - agent ?from - cell ?to - cell)
    :precondition (and (at ?a ?from) (free ?to) (adj ?from ?to))
    :effect (and (at ?a ?to) (not (at ?a ?from))
                 (free ?from) (not (free ?to)))))
