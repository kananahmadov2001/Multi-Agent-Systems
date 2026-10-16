(define (problem two-agents)
  (:domain conflict-test)
  (:objects
    a1 a2 - agent
    start1 start2 shared - cell)
  (:init
    (at a1 start1)
    (at a2 start2)
    (free shared))
  (:goal (and (at a1 shared) (at a2 shared))))
