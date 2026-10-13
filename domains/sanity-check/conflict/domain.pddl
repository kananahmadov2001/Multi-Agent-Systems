(define (domain conflict-test)
  (:requirements :strips :typing)
  (:types agent cell)
  (:predicates
    (at ?a - agent ?c - cell)
    (free ?c - cell))
  (:action move
    :parameters (?a - agent ?from - cell ?to - cell)
    :precondition (and (at ?a ?from) (free ?to))
    :effect (and (at ?a ?to) (not (at ?a ?from))
                 (free ?from) (not (free ?to)))))
