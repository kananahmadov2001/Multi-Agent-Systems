(define (problem two-agents-fixed)
  (:domain conflict-test)
  (:objects
    a1 a2 - agent
    start1 start2 shared other - cell)
  (:init
    (at a1 start1)
    (at a2 start2)
    (free shared)
    (free other))
  (:goal (and (at a1 shared) (at a2 other))))
