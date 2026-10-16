; Agent a2's private subproblem: cross the plus-shaped grid north -> south.
; Same grid as problem-a1. a2 starts at n, wants s. Both legs pass through c.
; a2 does not know a1 exists; its solo plan collides with a1's at cell c, t=1.
(define (problem a2-north-to-south)
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
  (:goal (at a2 s)))
