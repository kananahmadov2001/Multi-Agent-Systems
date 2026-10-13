(define (domain test)
  (:requirements :strips)
  (:predicates (at-a) (at-b))
  (:action move
    :parameters ()
    :precondition (at-a)
    :effect (and (at-b) (not (at-a)))))
