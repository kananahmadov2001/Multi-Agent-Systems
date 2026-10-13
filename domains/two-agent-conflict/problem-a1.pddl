; Agent a1's private subproblem: cross the plus-shaped grid west -> east.
; Cells:  n
;       w c e
;         s
; a1 starts at w, wants e. Both legs pass through center c.
(define (problem a1-west-to-east)
  (:domain grid-conflict)
  (:objects
    a1 - agent
    w c e n s - cell)
  (:init
    (at a1 w)
    (free c) (free e) (free n) (free s)
    (adj w c) (adj c w)
    (adj c e) (adj e c)
    (adj n c) (adj c n)
    (adj s c) (adj c s))
  (:goal (at a1 e)))
