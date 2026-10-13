; Negative control: a2 is already at its goal, so its plan is empty and it
; never leaves n. a1's path w -> c -> e never touches n, so the joint
; execution must be conflict-free.
(define (problem a2-stays-put)
  (:domain grid-conflict)
  (:objects
    a2 - agent
    w c e n s - cell)
  (:init
    (at a2 n)
    (free w) (free c) (free e) (free s)
    (adj w c) (adj c w)
    (adj c e) (adj e c)
    (adj n c) (adj c n)
    (adj s c) (adj c s))
  (:goal (at a2 n)))
