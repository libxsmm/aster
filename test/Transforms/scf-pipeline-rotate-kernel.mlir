// RUN: aster-opt %s --aster-scf-pipeline="rotate-kernel=true" | FileCheck %s

!vgpr = !amdgcn.vgpr

// CHECK-LABEL: func.func @rotate_2stage
// Peeled prologue: load cloned before loop.
//     CHECK:   amdgcn.test_inst{{.*}}{load}
// Rotated loop with additional iter_arg for crossing value.
//     CHECK:   scf.for
// Compute first (head).
//     CHECK:     amdgcn.test_inst{{.*}}{compute, sched.rotate_head}
// Load second (rest, shifted IV).
//     CHECK:     arith.addi
//     CHECK:     amdgcn.test_inst{{.*}}{load}
//     CHECK:     scf.yield
// Peeled epilogue: compute after loop.
//     CHECK:   amdgcn.test_inst{{.*}}{compute, sched.rotate_head}

func.func @rotate_2stage(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr
  %s1 = amdgcn.alloca : !vgpr
  %s_out = amdgcn.alloca : !vgpr
  %init = amdgcn.test_inst outs %s0 : (!vgpr) -> !vgpr

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr {
    %data = amdgcn.test_inst outs %s1
        {load, sched.stage = 0 : i32} : (!vgpr) -> !vgpr

    %new_acc = amdgcn.test_inst outs %s_out ins %acc, %data
        {compute, sched.stage = 1 : i32, sched.rotate_head}
        : (!vgpr, !vgpr, !vgpr) -> !vgpr

    scf.yield %new_acc : !vgpr
  }
  return
}


// CHECK-LABEL: func.func @no_rotate_head
// Pipeline prologue.
//     CHECK:   amdgcn.test_inst{{.*}}{load}
// Kernel loop: normal order (load first, compute second).
//     CHECK:   scf.for
//     CHECK:     amdgcn.test_inst{{.*}}{load}
//     CHECK:     amdgcn.test_inst{{.*}}{compute}
//     CHECK:     scf.yield
// Epilogue.
//     CHECK:   amdgcn.test_inst{{.*}}{compute}

func.func @no_rotate_head(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr
  %s1 = amdgcn.alloca : !vgpr
  %s_out = amdgcn.alloca : !vgpr
  %init = amdgcn.test_inst outs %s0 : (!vgpr) -> !vgpr

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr {
    %data = amdgcn.test_inst outs %s1
        {load, sched.stage = 0 : i32} : (!vgpr) -> !vgpr

    %new_acc = amdgcn.test_inst outs %s_out ins %acc, %data
        {compute, sched.stage = 1 : i32}
        : (!vgpr, !vgpr, !vgpr) -> !vgpr

    scf.yield %new_acc : !vgpr
  }
  return
}
