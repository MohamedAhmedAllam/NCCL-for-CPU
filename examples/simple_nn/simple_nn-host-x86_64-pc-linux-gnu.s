	.text
	.file	"simple_nn.cu"
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function __cxx_global_var_init
	.type	__cxx_global_var_init,@function
__cxx_global_var_init:                  # @__cxx_global_var_init
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	leaq	_ZStL8__ioinit(%rip), %rdi
	callq	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	_ZStL8__ioinit(%rip), %rsi
	leaq	__dso_handle(%rip), %rdx
	callq	__cxa_atexit@PLT
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	__cxx_global_var_init, .Lfunc_end0-__cxx_global_var_init
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype17Create_contiguousEi,"axG",@progbits,_ZNK3MPI8Datatype17Create_contiguousEi,comdat
	.weak	_ZNK3MPI8Datatype17Create_contiguousEi # -- Begin function _ZNK3MPI8Datatype17Create_contiguousEi
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype17Create_contiguousEi,@function
_ZNK3MPI8Datatype17Create_contiguousEi: # @_ZNK3MPI8Datatype17Create_contiguousEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movq	8(%rax), %rsi
	leaq	-32(%rbp), %rdx
	callq	MPI_Type_contiguous@PLT
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	-32(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_ZNK3MPI8Datatype17Create_contiguousEi, .Lfunc_end1-_ZNK3MPI8Datatype17Create_contiguousEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8DatatypeC2EP15ompi_datatype_t,"axG",@progbits,_ZN3MPI8DatatypeC2EP15ompi_datatype_t,comdat
	.weak	_ZN3MPI8DatatypeC2EP15ompi_datatype_t # -- Begin function _ZN3MPI8DatatypeC2EP15ompi_datatype_t
	.p2align	4, 0x90
	.type	_ZN3MPI8DatatypeC2EP15ompi_datatype_t,@function
_ZN3MPI8DatatypeC2EP15ompi_datatype_t:  # @_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	_ZTVN3MPI8DatatypeE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	_ZN3MPI8DatatypeC2EP15ompi_datatype_t, .Lfunc_end2-_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7Request4WaitERNS_6StatusE,"axG",@progbits,_ZN3MPI7Request4WaitERNS_6StatusE,comdat
	.weak	_ZN3MPI7Request4WaitERNS_6StatusE # -- Begin function _ZN3MPI7Request4WaitERNS_6StatusE
	.p2align	4, 0x90
	.type	_ZN3MPI7Request4WaitERNS_6StatusE,@function
_ZN3MPI7Request4WaitERNS_6StatusE:      # @_ZN3MPI7Request4WaitERNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	movq	-16(%rbp), %rsi
	addq	$8, %rsi
	callq	MPI_Wait@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	_ZN3MPI7Request4WaitERNS_6StatusE, .Lfunc_end3-_ZN3MPI7Request4WaitERNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8Prequest5StartEv,"axG",@progbits,_ZN3MPI8Prequest5StartEv,comdat
	.weak	_ZN3MPI8Prequest5StartEv        # -- Begin function _ZN3MPI8Prequest5StartEv
	.p2align	4, 0x90
	.type	_ZN3MPI8Prequest5StartEv,@function
_ZN3MPI8Prequest5StartEv:               # @_ZN3MPI8Prequest5StartEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Start@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end4:
	.size	_ZN3MPI8Prequest5StartEv, .Lfunc_end4-_ZN3MPI8Prequest5StartEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8Grequest8CompleteEv,"axG",@progbits,_ZN3MPI8Grequest8CompleteEv,comdat
	.weak	_ZN3MPI8Grequest8CompleteEv     # -- Begin function _ZN3MPI8Grequest8CompleteEv
	.p2align	4, 0x90
	.type	_ZN3MPI8Grequest8CompleteEv,@function
_ZN3MPI8Grequest8CompleteEv:            # @_ZN3MPI8Grequest8CompleteEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	callq	MPI_Grequest_complete@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end5:
	.size	_ZN3MPI8Grequest8CompleteEv, .Lfunc_end5-_ZN3MPI8Grequest8CompleteEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movq	8(%rax), %r9
	callq	MPI_Send@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end6:
	.size	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii, .Lfunc_end6-_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8DatatypecvP15ompi_datatype_tEv,"axG",@progbits,_ZNK3MPI8DatatypecvP15ompi_datatype_tEv,comdat
	.weak	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv # -- Begin function _ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	.p2align	4, 0x90
	.type	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv,@function
_ZNK3MPI8DatatypecvP15ompi_datatype_tEv: # @_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end7:
	.size	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv, .Lfunc_end7-_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE,"axG",@progbits,_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE,comdat
	.weak	_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE # -- Begin function _ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE,@function
_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE: # @_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI2OpcvP9ompi_op_tEv
	movq	-88(%rbp), %rdi                 # 8-byte Reload
	movq	-80(%rbp), %rsi                 # 8-byte Reload
	movl	-68(%rbp), %edx                 # 4-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %r8
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %r9
	callq	MPI_Scan@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end8:
	.size	_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE, .Lfunc_end8-_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI2OpcvP9ompi_op_tEv,"axG",@progbits,_ZNK3MPI2OpcvP9ompi_op_tEv,comdat
	.weak	_ZNK3MPI2OpcvP9ompi_op_tEv      # -- Begin function _ZNK3MPI2OpcvP9ompi_op_tEv
	.p2align	4, 0x90
	.type	_ZNK3MPI2OpcvP9ompi_op_tEv,@function
_ZNK3MPI2OpcvP9ompi_op_tEv:             # @_ZNK3MPI2OpcvP9ompi_op_tEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end9:
	.size	_ZNK3MPI2OpcvP9ompi_op_tEv, .Lfunc_end9-_ZNK3MPI2OpcvP9ompi_op_tEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm5CloneEv,"axG",@progbits,_ZNK3MPI8Cartcomm5CloneEv,comdat
	.weak	_ZNK3MPI8Cartcomm5CloneEv       # -- Begin function _ZNK3MPI8Cartcomm5CloneEv
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm5CloneEv,@function
_ZNK3MPI8Cartcomm5CloneEv:              # @_ZNK3MPI8Cartcomm5CloneEv
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-16(%rbp), %rsi
	movq	%rsi, -64(%rbp)                 # 8-byte Spill
	callq	MPI_Comm_dup@PLT
	movl	$16, %edi
	callq	_Znwm@PLT
	movq	-64(%rbp), %rsi                 # 8-byte Reload
	movq	%rax, %rdi
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
.Ltmp0:
	callq	_ZN3MPI8CartcommC2ERKP19ompi_communicator_t
.Ltmp1:
	jmp	.LBB10_1
.LBB10_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB10_2:
	.cfi_def_cfa %rbp, 16
.Ltmp2:
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZdlPv@PLT
# %bb.3:
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end10:
	.size	_ZNK3MPI8Cartcomm5CloneEv, .Lfunc_end10-_ZNK3MPI8Cartcomm5CloneEv
	.cfi_endproc
	.section	.gcc_except_table._ZNK3MPI8Cartcomm5CloneEv,"aG",@progbits,_ZNK3MPI8Cartcomm5CloneEv,comdat
	.p2align	2
GCC_except_table10:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Lfunc_end10-.Ltmp1            #   Call between .Ltmp1 and .Lfunc_end10
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2
                                        # -- End function
	.section	.text._ZN3MPI8CartcommC2ERKP19ompi_communicator_t,"axG",@progbits,_ZN3MPI8CartcommC2ERKP19ompi_communicator_t,comdat
	.weak	_ZN3MPI8CartcommC2ERKP19ompi_communicator_t # -- Begin function _ZN3MPI8CartcommC2ERKP19ompi_communicator_t
	.p2align	4, 0x90
	.type	_ZN3MPI8CartcommC2ERKP19ompi_communicator_t,@function
_ZN3MPI8CartcommC2ERKP19ompi_communicator_t: # @_ZN3MPI8CartcommC2ERKP19ompi_communicator_t
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception1
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI9IntracommC2Ev
	movq	-48(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI8CartcommE+16(%rip), %rcx
	movq	%rcx, (%rax)
	movl	$0, -20(%rbp)
.Ltmp3:
	callq	_ZN3MPI14Is_initializedEv
.Ltmp4:
	movb	%al, -37(%rbp)                  # 1-byte Spill
	jmp	.LBB11_1
.LBB11_1:
	movb	-37(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB11_2
	jmp	.LBB11_9
.LBB11_2:
	movq	-16(%rbp), %rax
	movq	ompi_mpi_comm_null@GOTPCREL(%rip), %rcx
	cmpq	%rcx, (%rax)
	je	.LBB11_9
# %bb.3:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdi
.Ltmp5:
	leaq	-20(%rbp), %rsi
	callq	MPI_Topo_test@PLT
.Ltmp6:
	jmp	.LBB11_4
.LBB11_4:
	cmpl	$1, -20(%rbp)
	jne	.LBB11_7
# %bb.5:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, 8(%rax)
	jmp	.LBB11_8
.LBB11_6:
.Ltmp7:
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZN3MPI9IntracommD2Ev
	jmp	.LBB11_11
.LBB11_7:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	ompi_mpi_comm_null@GOTPCREL(%rip), %rcx
	movq	%rcx, 8(%rax)
.LBB11_8:
	jmp	.LBB11_10
.LBB11_9:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, 8(%rax)
.LBB11_10:
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB11_11:
	.cfi_def_cfa %rbp, 16
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end11:
	.size	_ZN3MPI8CartcommC2ERKP19ompi_communicator_t, .Lfunc_end11-_ZN3MPI8CartcommC2ERKP19ompi_communicator_t
	.cfi_endproc
	.section	.gcc_except_table._ZN3MPI8CartcommC2ERKP19ompi_communicator_t,"aG",@progbits,_ZN3MPI8CartcommC2ERKP19ompi_communicator_t,comdat
	.p2align	2
GCC_except_table11:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp3-.Lfunc_begin1           #   Call between .Lfunc_begin1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin1           # >> Call Site 2 <<
	.uleb128 .Ltmp6-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp6
	.uleb128 .Ltmp7-.Lfunc_begin1           #     jumps to .Ltmp7
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin1           # >> Call Site 3 <<
	.uleb128 .Lfunc_end11-.Ltmp6            #   Call between .Ltmp6 and .Lfunc_end11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2
                                        # -- End function
	.section	.text._ZNK3MPI9Graphcomm5CloneEv,"axG",@progbits,_ZNK3MPI9Graphcomm5CloneEv,comdat
	.weak	_ZNK3MPI9Graphcomm5CloneEv      # -- Begin function _ZNK3MPI9Graphcomm5CloneEv
	.p2align	4, 0x90
	.type	_ZNK3MPI9Graphcomm5CloneEv,@function
_ZNK3MPI9Graphcomm5CloneEv:             # @_ZNK3MPI9Graphcomm5CloneEv
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception2
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-16(%rbp), %rsi
	movq	%rsi, -64(%rbp)                 # 8-byte Spill
	callq	MPI_Comm_dup@PLT
	movl	$16, %edi
	callq	_Znwm@PLT
	movq	-64(%rbp), %rsi                 # 8-byte Reload
	movq	%rax, %rdi
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
.Ltmp8:
	callq	_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t
.Ltmp9:
	jmp	.LBB12_1
.LBB12_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB12_2:
	.cfi_def_cfa %rbp, 16
.Ltmp10:
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZdlPv@PLT
# %bb.3:
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end12:
	.size	_ZNK3MPI9Graphcomm5CloneEv, .Lfunc_end12-_ZNK3MPI9Graphcomm5CloneEv
	.cfi_endproc
	.section	.gcc_except_table._ZNK3MPI9Graphcomm5CloneEv,"aG",@progbits,_ZNK3MPI9Graphcomm5CloneEv,comdat
	.p2align	2
GCC_except_table12:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Lfunc_begin2-.Lfunc_begin2    # >> Call Site 1 <<
	.uleb128 .Ltmp8-.Lfunc_begin2           #   Call between .Lfunc_begin2 and .Ltmp8
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp8-.Lfunc_begin2           # >> Call Site 2 <<
	.uleb128 .Ltmp9-.Ltmp8                  #   Call between .Ltmp8 and .Ltmp9
	.uleb128 .Ltmp10-.Lfunc_begin2          #     jumps to .Ltmp10
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin2           # >> Call Site 3 <<
	.uleb128 .Lfunc_end12-.Ltmp9            #   Call between .Ltmp9 and .Lfunc_end12
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end2:
	.p2align	2
                                        # -- End function
	.section	.text._ZN3MPI9GraphcommC2ERKP19ompi_communicator_t,"axG",@progbits,_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t,comdat
	.weak	_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t # -- Begin function _ZN3MPI9GraphcommC2ERKP19ompi_communicator_t
	.p2align	4, 0x90
	.type	_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t,@function
_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t: # @_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception3
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI9IntracommC2Ev
	movq	-48(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI9GraphcommE+16(%rip), %rcx
	movq	%rcx, (%rax)
	movl	$0, -20(%rbp)
.Ltmp11:
	callq	_ZN3MPI14Is_initializedEv
.Ltmp12:
	movb	%al, -37(%rbp)                  # 1-byte Spill
	jmp	.LBB13_1
.LBB13_1:
	movb	-37(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB13_2
	jmp	.LBB13_9
.LBB13_2:
	movq	-16(%rbp), %rax
	movq	ompi_mpi_comm_null@GOTPCREL(%rip), %rcx
	cmpq	%rcx, (%rax)
	je	.LBB13_9
# %bb.3:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdi
.Ltmp13:
	leaq	-20(%rbp), %rsi
	callq	MPI_Topo_test@PLT
.Ltmp14:
	jmp	.LBB13_4
.LBB13_4:
	cmpl	$2, -20(%rbp)
	jne	.LBB13_7
# %bb.5:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, 8(%rax)
	jmp	.LBB13_8
.LBB13_6:
.Ltmp15:
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZN3MPI9IntracommD2Ev
	jmp	.LBB13_11
.LBB13_7:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	ompi_mpi_comm_null@GOTPCREL(%rip), %rcx
	movq	%rcx, 8(%rax)
.LBB13_8:
	jmp	.LBB13_10
.LBB13_9:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, 8(%rax)
.LBB13_10:
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB13_11:
	.cfi_def_cfa %rbp, 16
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end13:
	.size	_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t, .Lfunc_end13-_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t
	.cfi_endproc
	.section	.gcc_except_table._ZN3MPI9GraphcommC2ERKP19ompi_communicator_t,"aG",@progbits,_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t,comdat
	.p2align	2
GCC_except_table13:
.Lexception3:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Lfunc_begin3-.Lfunc_begin3    # >> Call Site 1 <<
	.uleb128 .Ltmp11-.Lfunc_begin3          #   Call between .Lfunc_begin3 and .Ltmp11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin3          # >> Call Site 2 <<
	.uleb128 .Ltmp14-.Ltmp11                #   Call between .Ltmp11 and .Ltmp14
	.uleb128 .Ltmp15-.Lfunc_begin3          #     jumps to .Ltmp15
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp14-.Lfunc_begin3          # >> Call Site 3 <<
	.uleb128 .Lfunc_end13-.Ltmp14           #   Call between .Ltmp14 and .Lfunc_end13
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end3:
	.p2align	2
                                        # -- End function
	.section	.text._ZNK3MPI9Intercomm5CloneEv,"axG",@progbits,_ZNK3MPI9Intercomm5CloneEv,comdat
	.weak	_ZNK3MPI9Intercomm5CloneEv      # -- Begin function _ZNK3MPI9Intercomm5CloneEv
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intercomm5CloneEv,@function
_ZNK3MPI9Intercomm5CloneEv:             # @_ZNK3MPI9Intercomm5CloneEv
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception4
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-16(%rbp), %rsi
	callq	MPI_Comm_dup@PLT
	movl	$16, %edi
	callq	_Znwm@PLT
	movq	%rax, %rdi
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rsi
.Ltmp16:
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
.Ltmp17:
	jmp	.LBB14_1
.LBB14_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB14_2:
	.cfi_def_cfa %rbp, 16
.Ltmp18:
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZdlPv@PLT
# %bb.3:
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end14:
	.size	_ZNK3MPI9Intercomm5CloneEv, .Lfunc_end14-_ZNK3MPI9Intercomm5CloneEv
	.cfi_endproc
	.section	.gcc_except_table._ZNK3MPI9Intercomm5CloneEv,"aG",@progbits,_ZNK3MPI9Intercomm5CloneEv,comdat
	.p2align	2
GCC_except_table14:
.Lexception4:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Lfunc_begin4-.Lfunc_begin4    # >> Call Site 1 <<
	.uleb128 .Ltmp16-.Lfunc_begin4          #   Call between .Lfunc_begin4 and .Ltmp16
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp16-.Lfunc_begin4          # >> Call Site 2 <<
	.uleb128 .Ltmp17-.Ltmp16                #   Call between .Ltmp16 and .Ltmp17
	.uleb128 .Ltmp18-.Lfunc_begin4          #     jumps to .Ltmp18
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp17-.Lfunc_begin4          # >> Call Site 3 <<
	.uleb128 .Lfunc_end14-.Ltmp17           #   Call between .Ltmp17 and .Lfunc_end14
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end4:
	.p2align	2
                                        # -- End function
	.section	.text._ZN3MPI9IntercommC2EP19ompi_communicator_t,"axG",@progbits,_ZN3MPI9IntercommC2EP19ompi_communicator_t,comdat
	.weak	_ZN3MPI9IntercommC2EP19ompi_communicator_t # -- Begin function _ZN3MPI9IntercommC2EP19ompi_communicator_t
	.p2align	4, 0x90
	.type	_ZN3MPI9IntercommC2EP19ompi_communicator_t,@function
_ZN3MPI9IntercommC2EP19ompi_communicator_t: # @_ZN3MPI9IntercommC2EP19ompi_communicator_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -24(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rsi
	callq	_ZN3MPI4CommC2EP19ompi_communicator_t
	movq	-24(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI9IntercommE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end15:
	.size	_ZN3MPI9IntercommC2EP19ompi_communicator_t, .Lfunc_end15-_ZN3MPI9IntercommC2EP19ompi_communicator_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5Group8Get_sizeEv,"axG",@progbits,_ZNK3MPI5Group8Get_sizeEv,comdat
	.weak	_ZNK3MPI5Group8Get_sizeEv       # -- Begin function _ZNK3MPI5Group8Get_sizeEv
	.p2align	4, 0x90
	.type	_ZNK3MPI5Group8Get_sizeEv,@function
_ZNK3MPI5Group8Get_sizeEv:              # @_ZNK3MPI5Group8Get_sizeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Group_size@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end16:
	.size	_ZNK3MPI5Group8Get_sizeEv, .Lfunc_end16-_ZNK3MPI5Group8Get_sizeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI2OpD0Ev,"axG",@progbits,_ZN3MPI2OpD0Ev,comdat
	.weak	_ZN3MPI2OpD0Ev                  # -- Begin function _ZN3MPI2OpD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI2OpD0Ev,@function
_ZN3MPI2OpD0Ev:                         # @_ZN3MPI2OpD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI2OpD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end17:
	.size	_ZN3MPI2OpD0Ev, .Lfunc_end17-_ZN3MPI2OpD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI2OpD2Ev,"axG",@progbits,_ZN3MPI2OpD2Ev,comdat
	.weak	_ZN3MPI2OpD2Ev                  # -- Begin function _ZN3MPI2OpD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI2OpD2Ev,@function
_ZN3MPI2OpD2Ev:                         # @_ZN3MPI2OpD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end18:
	.size	_ZN3MPI2OpD2Ev, .Lfunc_end18-_ZN3MPI2OpD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI10Errhandler4FreeEv,"axG",@progbits,_ZN3MPI10Errhandler4FreeEv,comdat
	.weak	_ZN3MPI10Errhandler4FreeEv      # -- Begin function _ZN3MPI10Errhandler4FreeEv
	.p2align	4, 0x90
	.type	_ZN3MPI10Errhandler4FreeEv,@function
_ZN3MPI10Errhandler4FreeEv:             # @_ZN3MPI10Errhandler4FreeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Errhandler_free@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end19:
	.size	_ZN3MPI10Errhandler4FreeEv, .Lfunc_end19-_ZN3MPI10Errhandler4FreeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI6Status9Get_countERKNS_8DatatypeE,"axG",@progbits,_ZNK3MPI6Status9Get_countERKNS_8DatatypeE,comdat
	.weak	_ZNK3MPI6Status9Get_countERKNS_8DatatypeE # -- Begin function _ZNK3MPI6Status9Get_countERKNS_8DatatypeE
	.p2align	4, 0x90
	.type	_ZNK3MPI6Status9Get_countERKNS_8DatatypeE,@function
_ZNK3MPI6Status9Get_countERKNS_8DatatypeE: # @_ZNK3MPI6Status9Get_countERKNS_8DatatypeE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	addq	$8, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-32(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	leaq	-20(%rbp), %rdx
	callq	MPI_Get_count@PLT
	movl	-20(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end20:
	.size	_ZNK3MPI6Status9Get_countERKNS_8DatatypeE, .Lfunc_end20-_ZNK3MPI6Status9Get_countERKNS_8DatatypeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Info6DeleteEPKc,"axG",@progbits,_ZN3MPI4Info6DeleteEPKc,comdat
	.weak	_ZN3MPI4Info6DeleteEPKc         # -- Begin function _ZN3MPI4Info6DeleteEPKc
	.p2align	4, 0x90
	.type	_ZN3MPI4Info6DeleteEPKc,@function
_ZN3MPI4Info6DeleteEPKc:                # @_ZN3MPI4Info6DeleteEPKc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	callq	MPI_Info_delete@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end21:
	.size	_ZN3MPI4Info6DeleteEPKc, .Lfunc_end21-_ZN3MPI4Info6DeleteEPKc
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE,"axG",@progbits,_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE,comdat
	.weak	_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE # -- Begin function _ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE,@function
_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE: # @_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv
	movq	-24(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	callq	MPI_Win_set_errhandler@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end22:
	.size	_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE, .Lfunc_end22-_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv,"axG",@progbits,_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv,comdat
	.weak	_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv # -- Begin function _ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv
	.p2align	4, 0x90
	.type	_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv,@function
_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv: # @_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end23:
	.size	_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv, .Lfunc_end23-_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv
	.cfi_endproc
                                        # -- End function
	.text
	.globl	_Z21__device_stub__vecAddPfS_S_i # -- Begin function _Z21__device_stub__vecAddPfS_S_i
	.p2align	4, 0x90
	.type	_Z21__device_stub__vecAddPfS_S_i,@function
_Z21__device_stub__vecAddPfS_S_i:       # @_Z21__device_stub__vecAddPfS_S_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$160, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, -144(%rbp)
	leaq	-16(%rbp), %rax
	movq	%rax, -136(%rbp)
	leaq	-24(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-28(%rbp), %rax
	movq	%rax, -120(%rbp)
	leaq	-40(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	leaq	-64(%rbp), %rdx
	leaq	-72(%rbp), %rcx
	callq	__cudaPopCallConfiguration@PLT
	movq	-64(%rbp), %r10
	movq	-72(%rbp), %rax
	movq	-40(%rbp), %rcx
	movq	%rcx, -88(%rbp)
	movl	-32(%rbp), %ecx
	movl	%ecx, -80(%rbp)
	movq	-88(%rbp), %rsi
	movl	-80(%rbp), %edx
	movq	-56(%rbp), %rcx
	movq	%rcx, -104(%rbp)
	movl	-48(%rbp), %ecx
	movl	%ecx, -96(%rbp)
	movq	-104(%rbp), %rcx
	movl	-96(%rbp), %r8d
	leaq	_Z21__device_stub__vecAddPfS_S_i(%rip), %rdi
	leaq	-144(%rbp), %r9
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	cudaLaunchKernel@PLT
# %bb.1:
	addq	$160, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end24:
	.size	_Z21__device_stub__vecAddPfS_S_i, .Lfunc_end24-_Z21__device_stub__vecAddPfS_S_i
	.cfi_endproc
                                        # -- End function
	.globl	_Z27__device_stub__forward_passPfS_S_ii # -- Begin function _Z27__device_stub__forward_passPfS_S_ii
	.p2align	4, 0x90
	.type	_Z27__device_stub__forward_passPfS_S_ii,@function
_Z27__device_stub__forward_passPfS_S_ii: # @_Z27__device_stub__forward_passPfS_S_ii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$176, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movl	%r8d, -32(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, -160(%rbp)
	leaq	-16(%rbp), %rax
	movq	%rax, -152(%rbp)
	leaq	-24(%rbp), %rax
	movq	%rax, -144(%rbp)
	leaq	-28(%rbp), %rax
	movq	%rax, -136(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-48(%rbp), %rdi
	leaq	-64(%rbp), %rsi
	leaq	-72(%rbp), %rdx
	leaq	-80(%rbp), %rcx
	callq	__cudaPopCallConfiguration@PLT
	movq	-72(%rbp), %r10
	movq	-80(%rbp), %rax
	movq	-48(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	movl	-40(%rbp), %ecx
	movl	%ecx, -88(%rbp)
	movq	-96(%rbp), %rsi
	movl	-88(%rbp), %edx
	movq	-64(%rbp), %rcx
	movq	%rcx, -112(%rbp)
	movl	-56(%rbp), %ecx
	movl	%ecx, -104(%rbp)
	movq	-112(%rbp), %rcx
	movl	-104(%rbp), %r8d
	leaq	_Z27__device_stub__forward_passPfS_S_ii(%rip), %rdi
	leaq	-160(%rbp), %r9
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	cudaLaunchKernel@PLT
# %bb.1:
	addq	$176, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end25:
	.size	_Z27__device_stub__forward_passPfS_S_ii, .Lfunc_end25-_Z27__device_stub__forward_passPfS_S_ii
	.cfi_endproc
                                        # -- End function
	.globl	_Z28__device_stub__backward_passPfS_S_i # -- Begin function _Z28__device_stub__backward_passPfS_S_i
	.p2align	4, 0x90
	.type	_Z28__device_stub__backward_passPfS_S_i,@function
_Z28__device_stub__backward_passPfS_S_i: # @_Z28__device_stub__backward_passPfS_S_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$160, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, -144(%rbp)
	leaq	-16(%rbp), %rax
	movq	%rax, -136(%rbp)
	leaq	-24(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-28(%rbp), %rax
	movq	%rax, -120(%rbp)
	leaq	-40(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	leaq	-64(%rbp), %rdx
	leaq	-72(%rbp), %rcx
	callq	__cudaPopCallConfiguration@PLT
	movq	-64(%rbp), %r10
	movq	-72(%rbp), %rax
	movq	-40(%rbp), %rcx
	movq	%rcx, -88(%rbp)
	movl	-32(%rbp), %ecx
	movl	%ecx, -80(%rbp)
	movq	-88(%rbp), %rsi
	movl	-80(%rbp), %edx
	movq	-56(%rbp), %rcx
	movq	%rcx, -104(%rbp)
	movl	-48(%rbp), %ecx
	movl	%ecx, -96(%rbp)
	movq	-104(%rbp), %rcx
	movl	-96(%rbp), %r8d
	leaq	_Z28__device_stub__backward_passPfS_S_i(%rip), %rdi
	leaq	-144(%rbp), %r9
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	cudaLaunchKernel@PLT
# %bb.1:
	addq	$160, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end26:
	.size	_Z28__device_stub__backward_passPfS_S_i, .Lfunc_end26-_Z28__device_stub__backward_passPfS_S_i
	.cfi_endproc
                                        # -- End function
	.globl	_Z25__device_stub__calc_gradsPfS_S_ii # -- Begin function _Z25__device_stub__calc_gradsPfS_S_ii
	.p2align	4, 0x90
	.type	_Z25__device_stub__calc_gradsPfS_S_ii,@function
_Z25__device_stub__calc_gradsPfS_S_ii:  # @_Z25__device_stub__calc_gradsPfS_S_ii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$176, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movl	%r8d, -32(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, -160(%rbp)
	leaq	-16(%rbp), %rax
	movq	%rax, -152(%rbp)
	leaq	-24(%rbp), %rax
	movq	%rax, -144(%rbp)
	leaq	-28(%rbp), %rax
	movq	%rax, -136(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-48(%rbp), %rdi
	leaq	-64(%rbp), %rsi
	leaq	-72(%rbp), %rdx
	leaq	-80(%rbp), %rcx
	callq	__cudaPopCallConfiguration@PLT
	movq	-72(%rbp), %r10
	movq	-80(%rbp), %rax
	movq	-48(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	movl	-40(%rbp), %ecx
	movl	%ecx, -88(%rbp)
	movq	-96(%rbp), %rsi
	movl	-88(%rbp), %edx
	movq	-64(%rbp), %rcx
	movq	%rcx, -112(%rbp)
	movl	-56(%rbp), %ecx
	movl	%ecx, -104(%rbp)
	movq	-112(%rbp), %rcx
	movl	-104(%rbp), %r8d
	leaq	_Z25__device_stub__calc_gradsPfS_S_ii(%rip), %rdi
	leaq	-160(%rbp), %r9
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	cudaLaunchKernel@PLT
# %bb.1:
	addq	$176, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end27:
	.size	_Z25__device_stub__calc_gradsPfS_S_ii, .Lfunc_end27-_Z25__device_stub__calc_gradsPfS_S_ii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function main
.LCPI28_0:
	.long	0x3c23d70a                      # float 0.00999999977
.LCPI28_2:
	.long	0x3f800000                      # float 1
.LCPI28_3:
	.long	0x4f000000                      # float 2.14748365E+9
.LCPI28_4:
	.long	0xbf800000                      # float -1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3
.LCPI28_1:
	.quad	0x3fe0000000000000              # double 0.5
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$1840, %rsp                     # imm = 0x730
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$1024, -20(%rbp)                # imm = 0x400
	movl	$1000, -24(%rbp)                # imm = 0x3E8
	movss	.LCPI28_0(%rip), %xmm0          # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, -28(%rbp)
	movl	$0, -40(%rbp)
# %bb.1:
	leaq	-8(%rbp), %rdi
	leaq	-16(%rbp), %rsi
	callq	MPI_Init@PLT
	movl	%eax, -44(%rbp)
	cmpl	$0, -44(%rbp)
	je	.LBB28_3
# %bb.2:
	movl	-44(%rbp), %ecx
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$117, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_3:
	jmp	.LBB28_4
.LBB28_4:
	jmp	.LBB28_5
.LBB28_5:
	movq	ompi_mpi_comm_world@GOTPCREL(%rip), %rdi
	leaq	-32(%rbp), %rsi
	callq	MPI_Comm_rank@PLT
	movl	%eax, -48(%rbp)
	cmpl	$0, -48(%rbp)
	je	.LBB28_7
# %bb.6:
	movl	-48(%rbp), %ecx
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$118, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_7:
	jmp	.LBB28_8
.LBB28_8:
	jmp	.LBB28_9
.LBB28_9:
	movq	ompi_mpi_comm_world@GOTPCREL(%rip), %rdi
	leaq	-36(%rbp), %rsi
	callq	MPI_Comm_size@PLT
	movl	%eax, -52(%rbp)
	cmpl	$0, -52(%rbp)
	je	.LBB28_11
# %bb.10:
	movl	-52(%rbp), %ecx
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$119, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_11:
	jmp	.LBB28_12
.LBB28_12:
	movl	-36(%rbp), %eax
                                        # kill: def $rax killed $eax
	movq	%rsp, %rcx
	movq	%rcx, -64(%rbp)
	leaq	15(,%rax,8), %rdx
	andq	$-16, %rdx
	movq	%rsp, %rcx
	subq	%rdx, %rcx
	movq	%rcx, -1800(%rbp)               # 8-byte Spill
	movq	%rcx, %rsp
	movq	%rax, -72(%rbp)
	leaq	-1104(%rbp), %rdi
	movl	$1024, %esi                     # imm = 0x400
	callq	_ZL11getHostNamePci
	leaq	-1104(%rbp), %rdi
	callq	_ZL11getHostHashPKc
	movq	%rax, %rdx
	movq	-1800(%rbp), %rax               # 8-byte Reload
	movslq	-32(%rbp), %rcx
	movq	%rdx, (%rax,%rcx,8)
# %bb.13:
	movq	-1800(%rbp), %rcx               # 8-byte Reload
	movl	$1, %edi
	xorl	%esi, %esi
	movq	ompi_mpi_datatype_null@GOTPCREL(%rip), %rdx
	movl	$8, %r8d
	movq	ompi_mpi_byte@GOTPCREL(%rip), %r9
	movq	ompi_mpi_comm_world@GOTPCREL(%rip), %rax
	subq	$16, %rsp
	movq	%rax, (%rsp)
	callq	MPI_Allgather@PLT
	addq	$16, %rsp
	movl	%eax, -1108(%rbp)
	cmpl	$0, -1108(%rbp)
	je	.LBB28_15
# %bb.14:
	movl	-1108(%rbp), %ecx
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$126, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_15:
	jmp	.LBB28_16
.LBB28_16:
	jmp	.LBB28_17
.LBB28_17:
	movl	$0, -1112(%rbp)
.LBB28_18:                              # =>This Inner Loop Header: Depth=1
	movl	-1112(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jge	.LBB28_25
# %bb.19:                               #   in Loop: Header=BB28_18 Depth=1
	movl	-1112(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jne	.LBB28_21
# %bb.20:
	jmp	.LBB28_25
.LBB28_21:                              #   in Loop: Header=BB28_18 Depth=1
	movq	-1800(%rbp), %rcx               # 8-byte Reload
	movslq	-1112(%rbp), %rax
	movq	(%rcx,%rax,8), %rax
	movslq	-32(%rbp), %rdx
	cmpq	(%rcx,%rdx,8), %rax
	jne	.LBB28_23
# %bb.22:                               #   in Loop: Header=BB28_18 Depth=1
	movl	-40(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -40(%rbp)
.LBB28_23:                              #   in Loop: Header=BB28_18 Depth=1
	jmp	.LBB28_24
.LBB28_24:                              #   in Loop: Header=BB28_18 Depth=1
	movl	-1112(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1112(%rbp)
	jmp	.LBB28_18
.LBB28_25:
	jmp	.LBB28_26
.LBB28_26:
	leaq	-1240(%rbp), %rdi
	movl	$128, %esi
	movq	ompi_mpi_byte@GOTPCREL(%rip), %rdx
	xorl	%ecx, %ecx
	movq	ompi_mpi_comm_world@GOTPCREL(%rip), %r8
	callq	MPI_Bcast@PLT
	movl	%eax, -1260(%rbp)
	cmpl	$0, -1260(%rbp)
	je	.LBB28_28
# %bb.27:
	movl	-1260(%rbp), %ecx
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$138, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_28:
	jmp	.LBB28_29
.LBB28_29:
	jmp	.LBB28_30
.LBB28_30:
	jmp	.LBB28_31
.LBB28_31:
	leaq	-1256(%rbp), %rdi
	callq	cudaStreamCreate@PLT
	movl	%eax, -1264(%rbp)
	cmpl	$0, -1264(%rbp)
	je	.LBB28_33
# %bb.32:
	movl	-1264(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$143, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_33:
	jmp	.LBB28_34
.LBB28_34:
	jmp	.LBB28_35
.LBB28_35:
	jmp	.LBB28_36
.LBB28_36:
	movl	-36(%rbp), %eax
	movl	%eax, -1812(%rbp)               # 4-byte Spill
	leaq	-1400(%rbp), %rdi
	leaq	-1240(%rbp), %rsi
	movl	$128, %edx
	callq	memcpy@PLT
	movl	-1812(%rbp), %esi               # 4-byte Reload
	movl	-32(%rbp), %edx
	subq	$128, %rsp
	movups	-1288(%rbp), %xmm0
	movq	%rsp, %rax
	movq	%rax, -1808(%rbp)               # 8-byte Spill
	movups	%xmm0, 112(%rax)
	movups	-1304(%rbp), %xmm0
	movups	%xmm0, 96(%rax)
	movups	-1320(%rbp), %xmm0
	movups	%xmm0, 80(%rax)
	movups	-1336(%rbp), %xmm0
	movups	%xmm0, 64(%rax)
	movups	-1400(%rbp), %xmm0
	movups	-1384(%rbp), %xmm1
	movups	-1368(%rbp), %xmm2
	movups	-1352(%rbp), %xmm3
	movups	%xmm3, 48(%rax)
	movups	%xmm2, 32(%rax)
	movups	%xmm1, 16(%rax)
	movups	%xmm0, (%rax)
	leaq	-1248(%rbp), %rdi
	callq	ncclCommInitRank@PLT
	addq	$128, %rsp
	movl	%eax, -1268(%rbp)
	cmpl	$0, -1268(%rbp)
	je	.LBB28_38
# %bb.37:
	movl	-1268(%rbp), %edi
	callq	ncclGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.3(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$146, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_38:
	jmp	.LBB28_39
.LBB28_39:
	jmp	.LBB28_40
.LBB28_40:
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	movslq	%eax, %rdi
	shlq	$2, %rdi
	callq	malloc@PLT
	movq	%rax, -1408(%rbp)
	movslq	-24(%rbp), %rdi
	shlq	$2, %rdi
	callq	malloc@PLT
	movq	%rax, -1416(%rbp)
	movslq	-24(%rbp), %rdi
	shlq	$2, %rdi
	callq	malloc@PLT
	movq	%rax, -1424(%rbp)
	movslq	-20(%rbp), %rdi
	shlq	$2, %rdi
	callq	malloc@PLT
	movq	%rax, -1432(%rbp)
	movl	$0, -1436(%rbp)
.LBB28_41:                              # =>This Inner Loop Header: Depth=1
	movl	-1436(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.LBB28_44
# %bb.42:                               #   in Loop: Header=BB28_41 Depth=1
	callq	rand@PLT
	cvtsi2ss	%eax, %xmm0
	movss	.LCPI28_3(%rip), %xmm1          # xmm1 = mem[0],zero,zero,zero
	divss	%xmm1, %xmm0
	movss	.LCPI28_0(%rip), %xmm1          # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm0
	movq	-1416(%rbp), %rax
	movslq	-1436(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
# %bb.43:                               #   in Loop: Header=BB28_41 Depth=1
	movl	-1436(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1436(%rbp)
	jmp	.LBB28_41
.LBB28_44:
	xorl	%eax, %eax
	movl	%eax, %edi
	callq	time@PLT
	movslq	-32(%rbp), %rcx
	addq	%rcx, %rax
	movl	%eax, %edi
	callq	srand@PLT
	movl	$0, -1440(%rbp)
.LBB28_45:                              # =>This Inner Loop Header: Depth=1
	movl	-1440(%rbp), %eax
	movl	-20(%rbp), %ecx
	imull	-24(%rbp), %ecx
	cmpl	%ecx, %eax
	jge	.LBB28_48
# %bb.46:                               #   in Loop: Header=BB28_45 Depth=1
	callq	rand@PLT
	cvtsi2ss	%eax, %xmm0
	movss	.LCPI28_3(%rip), %xmm1          # xmm1 = mem[0],zero,zero,zero
	divss	%xmm1, %xmm0
	addss	%xmm0, %xmm0
	movss	.LCPI28_4(%rip), %xmm1          # xmm1 = mem[0],zero,zero,zero
	addss	%xmm1, %xmm0
	movq	-1408(%rbp), %rax
	movslq	-1440(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
# %bb.47:                               #   in Loop: Header=BB28_45 Depth=1
	movl	-1440(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1440(%rbp)
	jmp	.LBB28_45
.LBB28_48:
	movl	$0, -1444(%rbp)
.LBB28_49:                              # =>This Inner Loop Header: Depth=1
	movl	-1444(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.LBB28_52
# %bb.50:                               #   in Loop: Header=BB28_49 Depth=1
	callq	rand@PLT
	cvtsi2ss	%eax, %xmm0
	movss	.LCPI28_3(%rip), %xmm1          # xmm1 = mem[0],zero,zero,zero
	divss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm1
	movsd	.LCPI28_1(%rip), %xmm2          # xmm2 = mem[0],zero
	movss	.LCPI28_2(%rip), %xmm0          # xmm0 = mem[0],zero,zero,zero
	xorps	%xmm3, %xmm3
	movss	%xmm3, -1820(%rbp)              # 4-byte Spill
	ucomisd	%xmm2, %xmm1
	movss	%xmm0, -1816(%rbp)              # 4-byte Spill
	ja	.LBB28_184
# %bb.183:                              #   in Loop: Header=BB28_49 Depth=1
	movss	-1820(%rbp), %xmm0              # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, -1816(%rbp)              # 4-byte Spill
.LBB28_184:                             #   in Loop: Header=BB28_49 Depth=1
	movss	-1816(%rbp), %xmm0              # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movq	-1432(%rbp), %rax
	movslq	-1444(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
# %bb.51:                               #   in Loop: Header=BB28_49 Depth=1
	movl	-1444(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1444(%rbp)
	jmp	.LBB28_49
.LBB28_52:
	jmp	.LBB28_53
.LBB28_53:
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	movslq	%eax, %rsi
	shlq	$2, %rsi
	leaq	-1456(%rbp), %rdi
	callq	_ZL10cudaMallocIfE9cudaErrorPPT_m
	movl	%eax, -1500(%rbp)
	cmpl	$0, -1500(%rbp)
	je	.LBB28_55
# %bb.54:
	movl	-1500(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$172, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_55:
	jmp	.LBB28_56
.LBB28_56:
	jmp	.LBB28_57
.LBB28_57:
	jmp	.LBB28_58
.LBB28_58:
	movslq	-24(%rbp), %rsi
	shlq	$2, %rsi
	leaq	-1464(%rbp), %rdi
	callq	_ZL10cudaMallocIfE9cudaErrorPPT_m
	movl	%eax, -1504(%rbp)
	cmpl	$0, -1504(%rbp)
	je	.LBB28_60
# %bb.59:
	movl	-1504(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$173, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_60:
	jmp	.LBB28_61
.LBB28_61:
	jmp	.LBB28_62
.LBB28_62:
	jmp	.LBB28_63
.LBB28_63:
	movslq	-20(%rbp), %rsi
	shlq	$2, %rsi
	leaq	-1472(%rbp), %rdi
	callq	_ZL10cudaMallocIfE9cudaErrorPPT_m
	movl	%eax, -1508(%rbp)
	cmpl	$0, -1508(%rbp)
	je	.LBB28_65
# %bb.64:
	movl	-1508(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$174, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_65:
	jmp	.LBB28_66
.LBB28_66:
	jmp	.LBB28_67
.LBB28_67:
	jmp	.LBB28_68
.LBB28_68:
	movslq	-24(%rbp), %rsi
	shlq	$2, %rsi
	leaq	-1480(%rbp), %rdi
	callq	_ZL10cudaMallocIfE9cudaErrorPPT_m
	movl	%eax, -1512(%rbp)
	cmpl	$0, -1512(%rbp)
	je	.LBB28_70
# %bb.69:
	movl	-1512(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$175, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_70:
	jmp	.LBB28_71
.LBB28_71:
	jmp	.LBB28_72
.LBB28_72:
	jmp	.LBB28_73
.LBB28_73:
	movslq	-20(%rbp), %rsi
	shlq	$2, %rsi
	leaq	-1488(%rbp), %rdi
	callq	_ZL10cudaMallocIfE9cudaErrorPPT_m
	movl	%eax, -1516(%rbp)
	cmpl	$0, -1516(%rbp)
	je	.LBB28_75
# %bb.74:
	movl	-1516(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$176, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_75:
	jmp	.LBB28_76
.LBB28_76:
	jmp	.LBB28_77
.LBB28_77:
	jmp	.LBB28_78
.LBB28_78:
	movslq	-20(%rbp), %rsi
	shlq	$2, %rsi
	leaq	-1496(%rbp), %rdi
	callq	_ZL10cudaMallocIfE9cudaErrorPPT_m
	movl	%eax, -1520(%rbp)
	cmpl	$0, -1520(%rbp)
	je	.LBB28_80
# %bb.79:
	movl	-1520(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$177, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_80:
	jmp	.LBB28_81
.LBB28_81:
	jmp	.LBB28_82
.LBB28_82:
	jmp	.LBB28_83
.LBB28_83:
	movq	-1456(%rbp), %rdi
	movq	-1408(%rbp), %rsi
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	movslq	%eax, %rdx
	shlq	$2, %rdx
	movl	$1, %ecx
	callq	cudaMemcpy@PLT
	movl	%eax, -1524(%rbp)
	cmpl	$0, -1524(%rbp)
	je	.LBB28_85
# %bb.84:
	movl	-1524(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$181, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_85:
	jmp	.LBB28_86
.LBB28_86:
	jmp	.LBB28_87
.LBB28_87:
	jmp	.LBB28_88
.LBB28_88:
	movq	-1464(%rbp), %rdi
	movq	-1416(%rbp), %rsi
	movslq	-24(%rbp), %rdx
	shlq	$2, %rdx
	movl	$1, %ecx
	callq	cudaMemcpy@PLT
	movl	%eax, -1528(%rbp)
	cmpl	$0, -1528(%rbp)
	je	.LBB28_90
# %bb.89:
	movl	-1528(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$182, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_90:
	jmp	.LBB28_91
.LBB28_91:
	jmp	.LBB28_92
.LBB28_92:
	jmp	.LBB28_93
.LBB28_93:
	movq	-1472(%rbp), %rdi
	movq	-1432(%rbp), %rsi
	movslq	-20(%rbp), %rdx
	shlq	$2, %rdx
	movl	$1, %ecx
	callq	cudaMemcpy@PLT
	movl	%eax, -1532(%rbp)
	cmpl	$0, -1532(%rbp)
	je	.LBB28_95
# %bb.94:
	movl	-1532(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$183, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_95:
	jmp	.LBB28_96
.LBB28_96:
	jmp	.LBB28_97
.LBB28_97:
	movl	$1, -1536(%rbp)
	movl	$512, -1540(%rbp)               # imm = 0x200
	movl	-20(%rbp), %eax
	addl	-1540(%rbp), %eax
	subl	$1, %eax
	cltd
	idivl	-1540(%rbp)
	movl	%eax, -1544(%rbp)
	movl	$0, -1548(%rbp)
.LBB28_98:                              # =>This Inner Loop Header: Depth=1
	movl	-1548(%rbp), %eax
	cmpl	-1536(%rbp), %eax
	jge	.LBB28_137
# %bb.99:                               #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_100
.LBB28_100:                             #   in Loop: Header=BB28_98 Depth=1
	movq	-1480(%rbp), %rdi
	movslq	-24(%rbp), %rdx
	shlq	$2, %rdx
	xorl	%esi, %esi
	callq	cudaMemset@PLT
	movl	%eax, -1552(%rbp)
	cmpl	$0, -1552(%rbp)
	je	.LBB28_102
# %bb.101:
	movl	-1552(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$192, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_102:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_103
.LBB28_103:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_104
.LBB28_104:                             #   in Loop: Header=BB28_98 Depth=1
	movl	-1544(%rbp), %esi
	leaq	-1568(%rbp), %rdi
	movl	$1, %ecx
	movl	%ecx, %edx
	callq	_ZN4dim3C2Ejjj
	movl	-1540(%rbp), %esi
	leaq	-1584(%rbp), %rdi
	movl	$1, %ecx
	movl	%ecx, %edx
	callq	_ZN4dim3C2Ejjj
	movq	-1568(%rbp), %rax
	movq	%rax, -1600(%rbp)
	movl	-1560(%rbp), %eax
	movl	%eax, -1592(%rbp)
	movq	-1600(%rbp), %rdi
	movl	-1592(%rbp), %esi
	movq	-1584(%rbp), %rax
	movq	%rax, -1616(%rbp)
	movl	-1576(%rbp), %eax
	movl	%eax, -1608(%rbp)
	movq	-1616(%rbp), %rdx
	movl	-1608(%rbp), %ecx
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%r9, %r8
	callq	__cudaPushCallConfiguration@PLT
	cmpl	$0, %eax
	jne	.LBB28_106
# %bb.105:                              #   in Loop: Header=BB28_98 Depth=1
	movq	-1456(%rbp), %rdi
	movq	-1464(%rbp), %rsi
	movq	-1488(%rbp), %rdx
	movl	-20(%rbp), %ecx
	movl	-24(%rbp), %r8d
	callq	_Z27__device_stub__forward_passPfS_S_ii
.LBB28_106:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_107
.LBB28_107:                             #   in Loop: Header=BB28_98 Depth=1
	callq	cudaGetLastError@PLT
	movl	%eax, -1620(%rbp)
	cmpl	$0, -1620(%rbp)
	je	.LBB28_109
# %bb.108:
	movl	-1620(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$196, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_109:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_110
.LBB28_110:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_111
.LBB28_111:                             #   in Loop: Header=BB28_98 Depth=1
	movl	-1544(%rbp), %esi
	leaq	-1632(%rbp), %rdi
	movl	$1, %ecx
	movl	%ecx, %edx
	callq	_ZN4dim3C2Ejjj
	movl	-1540(%rbp), %esi
	leaq	-1648(%rbp), %rdi
	movl	$1, %ecx
	movl	%ecx, %edx
	callq	_ZN4dim3C2Ejjj
	movq	-1632(%rbp), %rax
	movq	%rax, -1664(%rbp)
	movl	-1624(%rbp), %eax
	movl	%eax, -1656(%rbp)
	movq	-1664(%rbp), %rdi
	movl	-1656(%rbp), %esi
	movq	-1648(%rbp), %rax
	movq	%rax, -1680(%rbp)
	movl	-1640(%rbp), %eax
	movl	%eax, -1672(%rbp)
	movq	-1680(%rbp), %rdx
	movl	-1672(%rbp), %ecx
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%r9, %r8
	callq	__cudaPushCallConfiguration@PLT
	cmpl	$0, %eax
	jne	.LBB28_113
# %bb.112:                              #   in Loop: Header=BB28_98 Depth=1
	movq	-1488(%rbp), %rdi
	movq	-1472(%rbp), %rsi
	movq	-1496(%rbp), %rdx
	movl	-20(%rbp), %ecx
	callq	_Z28__device_stub__backward_passPfS_S_i
.LBB28_113:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_114
.LBB28_114:                             #   in Loop: Header=BB28_98 Depth=1
	callq	cudaGetLastError@PLT
	movl	%eax, -1684(%rbp)
	cmpl	$0, -1684(%rbp)
	je	.LBB28_116
# %bb.115:
	movl	-1684(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$200, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_116:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_117
.LBB28_117:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_118
.LBB28_118:                             #   in Loop: Header=BB28_98 Depth=1
	movl	-24(%rbp), %eax
	addl	-1540(%rbp), %eax
	subl	$1, %eax
	cltd
	idivl	-1540(%rbp)
	movl	%eax, %esi
	leaq	-1696(%rbp), %rdi
	movl	$1, %ecx
	movl	%ecx, %edx
	callq	_ZN4dim3C2Ejjj
	movl	-1540(%rbp), %esi
	leaq	-1712(%rbp), %rdi
	movl	$1, %ecx
	movl	%ecx, %edx
	callq	_ZN4dim3C2Ejjj
	movq	-1696(%rbp), %rax
	movq	%rax, -1728(%rbp)
	movl	-1688(%rbp), %eax
	movl	%eax, -1720(%rbp)
	movq	-1728(%rbp), %rdi
	movl	-1720(%rbp), %esi
	movq	-1712(%rbp), %rax
	movq	%rax, -1744(%rbp)
	movl	-1704(%rbp), %eax
	movl	%eax, -1736(%rbp)
	movq	-1744(%rbp), %rdx
	movl	-1736(%rbp), %ecx
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%r9, %r8
	callq	__cudaPushCallConfiguration@PLT
	cmpl	$0, %eax
	jne	.LBB28_120
# %bb.119:                              #   in Loop: Header=BB28_98 Depth=1
	movq	-1456(%rbp), %rdi
	movq	-1496(%rbp), %rsi
	movq	-1480(%rbp), %rdx
	movl	-20(%rbp), %ecx
	movl	-24(%rbp), %r8d
	callq	_Z25__device_stub__calc_gradsPfS_S_ii
.LBB28_120:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_121
.LBB28_121:                             #   in Loop: Header=BB28_98 Depth=1
	callq	cudaGetLastError@PLT
	movl	%eax, -1748(%rbp)
	cmpl	$0, -1748(%rbp)
	je	.LBB28_123
# %bb.122:
	movl	-1748(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$204, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_123:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_124
.LBB28_124:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_125
.LBB28_125:                             #   in Loop: Header=BB28_98 Depth=1
	movl	-32(%rbp), %esi
	movq	-1480(%rbp), %rax
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movq	-1480(%rbp), %rax
	movq	%rax, -1832(%rbp)               # 8-byte Spill
	movl	-24(%rbp), %eax
	movl	$2, %ecx
	cltd
	idivl	%ecx
	movl	%eax, %ecx
	movq	-1832(%rbp), %rax               # 8-byte Reload
	movslq	%ecx, %rcx
	movss	(%rax,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	cvtss2sd	%xmm1, %xmm1
	movq	-1480(%rbp), %rax
	movl	-24(%rbp), %ecx
	subl	$1, %ecx
	movslq	%ecx, %rcx
	movss	(%rax,%rcx,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	cvtss2sd	%xmm2, %xmm2
	leaq	.L.str.4(%rip), %rdi
	movb	$3, %al
	callq	printf@PLT
# %bb.126:                              #   in Loop: Header=BB28_98 Depth=1
	movq	-1480(%rbp), %rdi
	movq	-1480(%rbp), %rsi
	movslq	-24(%rbp), %rdx
	movq	-1248(%rbp), %r9
	movq	-1256(%rbp), %rax
	movl	$7, %ecx
	xorl	%r8d, %r8d
	subq	$16, %rsp
	movq	%rax, (%rsp)
	callq	ncclAllReduce@PLT
	addq	$16, %rsp
	movl	%eax, -1752(%rbp)
	cmpl	$0, -1752(%rbp)
	je	.LBB28_128
# %bb.127:
	movl	-1752(%rbp), %edi
	callq	ncclGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.3(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$209, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_128:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_129
.LBB28_129:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_130
.LBB28_130:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_131
.LBB28_131:                             #   in Loop: Header=BB28_98 Depth=1
	movq	-1256(%rbp), %rdi
	callq	cudaStreamSynchronize@PLT
	movl	%eax, -1756(%rbp)
	cmpl	$0, -1756(%rbp)
	je	.LBB28_133
# %bb.132:
	movl	-1756(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$210, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_133:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_134
.LBB28_134:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_135
.LBB28_135:                             #   in Loop: Header=BB28_98 Depth=1
	jmp	.LBB28_136
.LBB28_136:                             #   in Loop: Header=BB28_98 Depth=1
	movl	-1548(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1548(%rbp)
	jmp	.LBB28_98
.LBB28_137:
	jmp	.LBB28_138
.LBB28_138:
	movq	-1424(%rbp), %rdi
	movq	-1480(%rbp), %rsi
	movslq	-24(%rbp), %rdx
	shlq	$2, %rdx
	movl	$2, %ecx
	callq	cudaMemcpy@PLT
	movl	%eax, -1760(%rbp)
	cmpl	$0, -1760(%rbp)
	je	.LBB28_140
# %bb.139:
	movl	-1760(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$215, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_140:
	jmp	.LBB28_141
.LBB28_141:
	jmp	.LBB28_142
.LBB28_142:
	movl	-32(%rbp), %esi
	movq	-1480(%rbp), %rax
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movq	-1480(%rbp), %rax
	movq	%rax, -1840(%rbp)               # 8-byte Spill
	movl	-24(%rbp), %eax
	movl	$2, %ecx
	cltd
	idivl	%ecx
	movl	%eax, %ecx
	movq	-1840(%rbp), %rax               # 8-byte Reload
	movslq	%ecx, %rcx
	movss	(%rax,%rcx,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	cvtss2sd	%xmm1, %xmm1
	movq	-1480(%rbp), %rax
	movl	-24(%rbp), %ecx
	subl	$1, %ecx
	movslq	%ecx, %rcx
	movss	(%rax,%rcx,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	cvtss2sd	%xmm2, %xmm2
	leaq	.L.str.5(%rip), %rdi
	movb	$3, %al
	callq	printf@PLT
# %bb.143:
	movq	-1456(%rbp), %rdi
	callq	cudaFree@PLT
	movl	%eax, -1764(%rbp)
	cmpl	$0, -1764(%rbp)
	je	.LBB28_145
# %bb.144:
	movl	-1764(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$219, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_145:
	jmp	.LBB28_146
.LBB28_146:
	jmp	.LBB28_147
.LBB28_147:
	jmp	.LBB28_148
.LBB28_148:
	movq	-1464(%rbp), %rdi
	callq	cudaFree@PLT
	movl	%eax, -1768(%rbp)
	cmpl	$0, -1768(%rbp)
	je	.LBB28_150
# %bb.149:
	movl	-1768(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$220, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_150:
	jmp	.LBB28_151
.LBB28_151:
	jmp	.LBB28_152
.LBB28_152:
	jmp	.LBB28_153
.LBB28_153:
	movq	-1480(%rbp), %rdi
	callq	cudaFree@PLT
	movl	%eax, -1772(%rbp)
	cmpl	$0, -1772(%rbp)
	je	.LBB28_155
# %bb.154:
	movl	-1772(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$221, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_155:
	jmp	.LBB28_156
.LBB28_156:
	jmp	.LBB28_157
.LBB28_157:
	jmp	.LBB28_158
.LBB28_158:
	movq	-1472(%rbp), %rdi
	callq	cudaFree@PLT
	movl	%eax, -1776(%rbp)
	cmpl	$0, -1776(%rbp)
	je	.LBB28_160
# %bb.159:
	movl	-1776(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$222, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_160:
	jmp	.LBB28_161
.LBB28_161:
	jmp	.LBB28_162
.LBB28_162:
	jmp	.LBB28_163
.LBB28_163:
	movq	-1488(%rbp), %rdi
	callq	cudaFree@PLT
	movl	%eax, -1780(%rbp)
	cmpl	$0, -1780(%rbp)
	je	.LBB28_165
# %bb.164:
	movl	-1780(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$223, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_165:
	jmp	.LBB28_166
.LBB28_166:
	jmp	.LBB28_167
.LBB28_167:
	jmp	.LBB28_168
.LBB28_168:
	movq	-1496(%rbp), %rdi
	callq	cudaFree@PLT
	movl	%eax, -1784(%rbp)
	cmpl	$0, -1784(%rbp)
	je	.LBB28_170
# %bb.169:
	movl	-1784(%rbp), %edi
	callq	cudaGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.2(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$224, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_170:
	jmp	.LBB28_171
.LBB28_171:
	jmp	.LBB28_172
.LBB28_172:
	movq	-1408(%rbp), %rdi
	callq	free@PLT
	movq	-1416(%rbp), %rdi
	callq	free@PLT
	movq	-1424(%rbp), %rdi
	callq	free@PLT
	movq	-1432(%rbp), %rdi
	callq	free@PLT
# %bb.173:
	movq	-1248(%rbp), %rdi
	callq	ncclCommDestroy@PLT
	movl	%eax, -1788(%rbp)
	cmpl	$0, -1788(%rbp)
	je	.LBB28_175
# %bb.174:
	movl	-1788(%rbp), %edi
	callq	ncclGetErrorString@PLT
	movq	%rax, %rcx
	leaq	.L.str.3(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$233, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_175:
	jmp	.LBB28_176
.LBB28_176:
	jmp	.LBB28_177
.LBB28_177:
	jmp	.LBB28_178
.LBB28_178:
	callq	MPI_Finalize@PLT
	movl	%eax, -1792(%rbp)
	cmpl	$0, -1792(%rbp)
	je	.LBB28_180
# %bb.179:
	movl	-1792(%rbp), %ecx
	leaq	.L.str(%rip), %rdi
	leaq	.L.str.1(%rip), %rsi
	movl	$236, %edx
	movb	$0, %al
	callq	printf@PLT
	movl	$1, %edi
	callq	exit@PLT
.LBB28_180:
	jmp	.LBB28_181
.LBB28_181:
	jmp	.LBB28_182
.LBB28_182:
	movl	-32(%rbp), %esi
	leaq	.L.str.6(%rip), %rdi
	movb	$0, %al
	callq	printf@PLT
	movl	$0, -4(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, %rsp
	movl	-4(%rbp), %eax
	movq	%rbp, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end28:
	.size	main, .Lfunc_end28-main
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZL11getHostNamePci
	.type	_ZL11getHostNamePci,@function
_ZL11getHostNamePci:                    # @_ZL11getHostNamePci
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movslq	-12(%rbp), %rsi
	callq	gethostname@PLT
	movl	$0, -16(%rbp)
.LBB29_1:                               # =>This Inner Loop Header: Depth=1
	movl	-16(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jge	.LBB29_6
# %bb.2:                                #   in Loop: Header=BB29_1 Depth=1
	movq	-8(%rbp), %rax
	movslq	-16(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	cmpl	$46, %eax
	jne	.LBB29_4
# %bb.3:
	movq	-8(%rbp), %rax
	movslq	-16(%rbp), %rcx
	movb	$0, (%rax,%rcx)
	jmp	.LBB29_6
.LBB29_4:                               #   in Loop: Header=BB29_1 Depth=1
	jmp	.LBB29_5
.LBB29_5:                               #   in Loop: Header=BB29_1 Depth=1
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -16(%rbp)
	jmp	.LBB29_1
.LBB29_6:
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end29:
	.size	_ZL11getHostNamePci, .Lfunc_end29-_ZL11getHostNamePci
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZL11getHostHashPKc
	.type	_ZL11getHostHashPKc,@function
_ZL11getHostHashPKc:                    # @_ZL11getHostHashPKc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	$5381, -16(%rbp)                # imm = 0x1505
	movl	$0, -20(%rbp)
.LBB30_1:                               # =>This Inner Loop Header: Depth=1
	movq	-8(%rbp), %rax
	movslq	-20(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	cmpl	$0, %eax
	je	.LBB30_4
# %bb.2:                                #   in Loop: Header=BB30_1 Depth=1
	movq	-16(%rbp), %rax
	shlq	$5, %rax
	addq	-16(%rbp), %rax
	movq	-8(%rbp), %rcx
	movslq	-20(%rbp), %rdx
	movsbq	(%rcx,%rdx), %rcx
	xorq	%rcx, %rax
	movq	%rax, -16(%rbp)
# %bb.3:                                #   in Loop: Header=BB30_1 Depth=1
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	jmp	.LBB30_1
.LBB30_4:
	movq	-16(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end30:
	.size	_ZL11getHostHashPKc, .Lfunc_end30-_ZL11getHostHashPKc
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function _ZL10cudaMallocIfE9cudaErrorPPT_m
	.type	_ZL10cudaMallocIfE9cudaErrorPPT_m,@function
_ZL10cudaMallocIfE9cudaErrorPPT_m:      # @_ZL10cudaMallocIfE9cudaErrorPPT_m
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	cudaMalloc@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end31:
	.size	_ZL10cudaMallocIfE9cudaErrorPPT_m, .Lfunc_end31-_ZL10cudaMallocIfE9cudaErrorPPT_m
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN4dim3C2Ejjj,"axG",@progbits,_ZN4dim3C2Ejjj,comdat
	.weak	_ZN4dim3C2Ejjj                  # -- Begin function _ZN4dim3C2Ejjj
	.p2align	4, 0x90
	.type	_ZN4dim3C2Ejjj,@function
_ZN4dim3C2Ejjj:                         # @_ZN4dim3C2Ejjj
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	movl	-16(%rbp), %ecx
	movl	%ecx, 4(%rax)
	movl	-20(%rbp), %ecx
	movl	%ecx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end32:
	.size	_ZN4dim3C2Ejjj, .Lfunc_end32-_ZN4dim3C2Ejjj
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8DatatypeD2Ev,"axG",@progbits,_ZN3MPI8DatatypeD2Ev,comdat
	.weak	_ZN3MPI8DatatypeD2Ev            # -- Begin function _ZN3MPI8DatatypeD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8DatatypeD2Ev,@function
_ZN3MPI8DatatypeD2Ev:                   # @_ZN3MPI8DatatypeD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end33:
	.size	_ZN3MPI8DatatypeD2Ev, .Lfunc_end33-_ZN3MPI8DatatypeD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8DatatypeD0Ev,"axG",@progbits,_ZN3MPI8DatatypeD0Ev,comdat
	.weak	_ZN3MPI8DatatypeD0Ev            # -- Begin function _ZN3MPI8DatatypeD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8DatatypeD0Ev,@function
_ZN3MPI8DatatypeD0Ev:                   # @_ZN3MPI8DatatypeD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI8DatatypeD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end34:
	.size	_ZN3MPI8DatatypeD0Ev, .Lfunc_end34-_ZN3MPI8DatatypeD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype13Create_vectorEiii,"axG",@progbits,_ZNK3MPI8Datatype13Create_vectorEiii,comdat
	.weak	_ZNK3MPI8Datatype13Create_vectorEiii # -- Begin function _ZNK3MPI8Datatype13Create_vectorEiii
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype13Create_vectorEiii,@function
_ZNK3MPI8Datatype13Create_vectorEiii:   # @_ZNK3MPI8Datatype13Create_vectorEiii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movl	%ecx, -24(%rbp)
	movl	%r8d, -28(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movl	-24(%rbp), %esi
	movl	-28(%rbp), %edx
	movq	8(%rax), %rcx
	leaq	-40(%rbp), %r8
	callq	MPI_Type_vector@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end35:
	.size	_ZNK3MPI8Datatype13Create_vectorEiii, .Lfunc_end35-_ZNK3MPI8Datatype13Create_vectorEiii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype14Create_indexedEiPKiS2_,"axG",@progbits,_ZNK3MPI8Datatype14Create_indexedEiPKiS2_,comdat
	.weak	_ZNK3MPI8Datatype14Create_indexedEiPKiS2_ # -- Begin function _ZNK3MPI8Datatype14Create_indexedEiPKiS2_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype14Create_indexedEiPKiS2_,@function
_ZNK3MPI8Datatype14Create_indexedEiPKiS2_: # @_ZNK3MPI8Datatype14Create_indexedEiPKiS2_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -64(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rdx
	movq	8(%rax), %rcx
	leaq	-48(%rbp), %r8
	callq	MPI_Type_indexed@PLT
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movq	-48(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end36:
	.size	_ZNK3MPI8Datatype14Create_indexedEiPKiS2_, .Lfunc_end36-_ZNK3MPI8Datatype14Create_indexedEiPKiS2_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype15Create_hindexedEiPKiPKl,"axG",@progbits,_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl,comdat
	.weak	_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl # -- Begin function _ZNK3MPI8Datatype15Create_hindexedEiPKiPKl
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl,@function
_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl: # @_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -64(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rdx
	movq	8(%rax), %rcx
	leaq	-48(%rbp), %r8
	callq	MPI_Type_create_hindexed@PLT
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movq	-48(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end37:
	.size	_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl, .Lfunc_end37-_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype14Create_hvectorEiil,"axG",@progbits,_ZNK3MPI8Datatype14Create_hvectorEiil,comdat
	.weak	_ZNK3MPI8Datatype14Create_hvectorEiil # -- Begin function _ZNK3MPI8Datatype14Create_hvectorEiil
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype14Create_hvectorEiil,@function
_ZNK3MPI8Datatype14Create_hvectorEiil:  # @_ZNK3MPI8Datatype14Create_hvectorEiil
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movl	%ecx, -24(%rbp)
	movq	%r8, -32(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movl	-24(%rbp), %esi
	movq	-32(%rbp), %rdx
	movq	8(%rax), %rcx
	leaq	-40(%rbp), %r8
	callq	MPI_Type_create_hvector@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end38:
	.size	_ZNK3MPI8Datatype14Create_hvectorEiil, .Lfunc_end38-_ZNK3MPI8Datatype14Create_hvectorEiil
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype20Create_indexed_blockEiiPKi,"axG",@progbits,_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi,comdat
	.weak	_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi # -- Begin function _ZNK3MPI8Datatype20Create_indexed_blockEiiPKi
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi,@function
_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi: # @_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movl	%ecx, -24(%rbp)
	movq	%r8, -32(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movl	-24(%rbp), %esi
	movq	-32(%rbp), %rdx
	movq	8(%rax), %rcx
	leaq	-40(%rbp), %r8
	callq	MPI_Type_create_indexed_block@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end39:
	.size	_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi, .Lfunc_end39-_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype14Create_resizedEll,"axG",@progbits,_ZNK3MPI8Datatype14Create_resizedEll,comdat
	.weak	_ZNK3MPI8Datatype14Create_resizedEll # -- Begin function _ZNK3MPI8Datatype14Create_resizedEll
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype14Create_resizedEll,@function
_ZNK3MPI8Datatype14Create_resizedEll:   # @_ZNK3MPI8Datatype14Create_resizedEll
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rcx
	callq	MPI_Type_create_resized@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end40:
	.size	_ZNK3MPI8Datatype14Create_resizedEll, .Lfunc_end40-_ZNK3MPI8Datatype14Create_resizedEll
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype8Get_sizeEv,"axG",@progbits,_ZNK3MPI8Datatype8Get_sizeEv,comdat
	.weak	_ZNK3MPI8Datatype8Get_sizeEv    # -- Begin function _ZNK3MPI8Datatype8Get_sizeEv
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype8Get_sizeEv,@function
_ZNK3MPI8Datatype8Get_sizeEv:           # @_ZNK3MPI8Datatype8Get_sizeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Type_size@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end41:
	.size	_ZNK3MPI8Datatype8Get_sizeEv, .Lfunc_end41-_ZNK3MPI8Datatype8Get_sizeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype10Get_extentERlS1_,"axG",@progbits,_ZNK3MPI8Datatype10Get_extentERlS1_,comdat
	.weak	_ZNK3MPI8Datatype10Get_extentERlS1_ # -- Begin function _ZNK3MPI8Datatype10Get_extentERlS1_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype10Get_extentERlS1_,@function
_ZNK3MPI8Datatype10Get_extentERlS1_:    # @_ZNK3MPI8Datatype10Get_extentERlS1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Type_get_extent@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end42:
	.size	_ZNK3MPI8Datatype10Get_extentERlS1_, .Lfunc_end42-_ZNK3MPI8Datatype10Get_extentERlS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype15Get_true_extentERlS1_,"axG",@progbits,_ZNK3MPI8Datatype15Get_true_extentERlS1_,comdat
	.weak	_ZNK3MPI8Datatype15Get_true_extentERlS1_ # -- Begin function _ZNK3MPI8Datatype15Get_true_extentERlS1_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype15Get_true_extentERlS1_,@function
_ZNK3MPI8Datatype15Get_true_extentERlS1_: # @_ZNK3MPI8Datatype15Get_true_extentERlS1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Type_get_true_extent@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end43:
	.size	_ZNK3MPI8Datatype15Get_true_extentERlS1_, .Lfunc_end43-_ZNK3MPI8Datatype15Get_true_extentERlS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8Datatype6CommitEv,"axG",@progbits,_ZN3MPI8Datatype6CommitEv,comdat
	.weak	_ZN3MPI8Datatype6CommitEv       # -- Begin function _ZN3MPI8Datatype6CommitEv
	.p2align	4, 0x90
	.type	_ZN3MPI8Datatype6CommitEv,@function
_ZN3MPI8Datatype6CommitEv:              # @_ZN3MPI8Datatype6CommitEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Type_commit@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end44:
	.size	_ZN3MPI8Datatype6CommitEv, .Lfunc_end44-_ZN3MPI8Datatype6CommitEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE,"axG",@progbits,_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE,comdat
	.weak	_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE # -- Begin function _ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE,@function
_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE: # @_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movq	%rcx, -96(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %ecx
	movl	%ecx, -84(%rbp)                 # 4-byte Spill
	movq	8(%rax), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	16(%rbp), %rdi
	callq	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movl	-60(%rbp), %r8d                 # 4-byte Reload
	movq	-56(%rbp), %r9                  # 8-byte Reload
	movq	%rax, (%rsp)
	callq	MPI_Pack@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end45:
	.size	_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE, .Lfunc_end45-_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE,"axG",@progbits,_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE,comdat
	.weak	_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE # -- Begin function _ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE,@function
_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE: # @_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movq	%rcx, -96(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %ecx
	movl	%ecx, -84(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rcx
	movq	%rcx, -80(%rbp)                 # 8-byte Spill
	movq	-32(%rbp), %rcx
	movq	%rcx, -72(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %ecx
	movl	%ecx, -60(%rbp)                 # 4-byte Spill
	movq	8(%rax), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	16(%rbp), %rdi
	callq	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movl	-60(%rbp), %r8d                 # 4-byte Reload
	movq	-56(%rbp), %r9                  # 8-byte Reload
	movq	%rax, (%rsp)
	callq	MPI_Unpack@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end46:
	.size	_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE, .Lfunc_end46-_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE,"axG",@progbits,_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE,comdat
	.weak	_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE # -- Begin function _ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE,@function
_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE: # @_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, -44(%rbp)                 # 4-byte Spill
	movq	8(%rax), %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rdi
	callq	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	movl	-44(%rbp), %edi                 # 4-byte Reload
	movq	-40(%rbp), %rsi                 # 8-byte Reload
	movq	%rax, %rdx
	leaq	-28(%rbp), %rcx
	callq	MPI_Pack_size@PLT
	movl	-28(%rbp), %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end47:
	.size	_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE, .Lfunc_end47-_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl,"axG",@progbits,_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl,comdat
	.weak	_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl # -- Begin function _ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl,@function
_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl: # @_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdi
	movq	-24(%rbp), %rsi
	movl	-28(%rbp), %edx
	movq	8(%rax), %rcx
	movq	-40(%rbp), %r8
	movq	-48(%rbp), %r9
	movq	16(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Pack_external@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end48:
	.size	_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl, .Lfunc_end48-_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype18Pack_external_sizeEPKci,"axG",@progbits,_ZNK3MPI8Datatype18Pack_external_sizeEPKci,comdat
	.weak	_ZNK3MPI8Datatype18Pack_external_sizeEPKci # -- Begin function _ZNK3MPI8Datatype18Pack_external_sizeEPKci
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype18Pack_external_sizeEPKci,@function
_ZNK3MPI8Datatype18Pack_external_sizeEPKci: # @_ZNK3MPI8Datatype18Pack_external_sizeEPKci
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdi
	movl	-20(%rbp), %esi
	movq	8(%rax), %rdx
	leaq	-32(%rbp), %rcx
	callq	MPI_Pack_external_size@PLT
	movq	-32(%rbp), %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end49:
	.size	_ZNK3MPI8Datatype18Pack_external_sizeEPKci, .Lfunc_end49-_ZNK3MPI8Datatype18Pack_external_sizeEPKci
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi,"axG",@progbits,_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi,comdat
	.weak	_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi # -- Begin function _ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi,@function
_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi: # @_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdi
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %r8
	movl	16(%rbp), %r9d
	movq	8(%rax), %rax
	movq	%rax, (%rsp)
	callq	MPI_Unpack_external@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end50:
	.size	_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi, .Lfunc_end50-_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i,"axG",@progbits,_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i,comdat
	.weak	_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i # -- Begin function _ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i,@function
_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i: # @_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rdx
	movq	-48(%rbp), %rcx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Type_create_subarray@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end51:
	.size	_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i, .Lfunc_end51-_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i,"axG",@progbits,_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i,comdat
	.weak	_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i # -- Begin function _ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i,@function
_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i: # @_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%rbx
	subq	$104, %rsp
	.cfi_offset %rbx, -24
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	40(%rbp), %eax
	movq	32(%rbp), %rax
	movq	24(%rbp), %rax
	movq	16(%rbp), %rax
	movq	%rdi, -16(%rbp)
	movq	%rsi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movl	%ecx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edi
	movl	-32(%rbp), %esi
	movl	-36(%rbp), %edx
	movq	-48(%rbp), %rcx
	movq	16(%rbp), %r8
	movq	24(%rbp), %r9
	movq	32(%rbp), %rbx
	movl	40(%rbp), %r11d
	movq	8(%rax), %r10
	leaq	-56(%rbp), %rax
	movq	%rbx, (%rsp)
	movl	%r11d, 8(%rsp)
	movq	%r10, 16(%rsp)
	movq	%rax, 24(%rsp)
	callq	MPI_Type_create_darray@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$104, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end52:
	.size	_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i, .Lfunc_end52-_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype3DupEv,"axG",@progbits,_ZNK3MPI8Datatype3DupEv,comdat
	.weak	_ZNK3MPI8Datatype3DupEv         # -- Begin function _ZNK3MPI8Datatype3DupEv
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype3DupEv,@function
_ZNK3MPI8Datatype3DupEv:                # @_ZNK3MPI8Datatype3DupEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Type_dup@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI8DatatypeC2EP15ompi_datatype_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end53:
	.size	_ZNK3MPI8Datatype3DupEv, .Lfunc_end53-_ZNK3MPI8Datatype3DupEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8Datatype11Delete_attrEi,"axG",@progbits,_ZN3MPI8Datatype11Delete_attrEi,comdat
	.weak	_ZN3MPI8Datatype11Delete_attrEi # -- Begin function _ZN3MPI8Datatype11Delete_attrEi
	.p2align	4, 0x90
	.type	_ZN3MPI8Datatype11Delete_attrEi,@function
_ZN3MPI8Datatype11Delete_attrEi:        # @_ZN3MPI8Datatype11Delete_attrEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	callq	MPI_Type_delete_attr@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end54:
	.size	_ZN3MPI8Datatype11Delete_attrEi, .Lfunc_end54-_ZN3MPI8Datatype11Delete_attrEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype8Get_attrEiPv,"axG",@progbits,_ZNK3MPI8Datatype8Get_attrEiPv,comdat
	.weak	_ZNK3MPI8Datatype8Get_attrEiPv  # -- Begin function _ZNK3MPI8Datatype8Get_attrEiPv
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype8Get_attrEiPv,@function
_ZNK3MPI8Datatype8Get_attrEiPv:         # @_ZNK3MPI8Datatype8Get_attrEiPv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rcx
	callq	MPI_Type_get_attr@PLT
	cmpl	$0, -28(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end55:
	.size	_ZNK3MPI8Datatype8Get_attrEiPv, .Lfunc_end55-_ZNK3MPI8Datatype8Get_attrEiPv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_,"axG",@progbits,_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_,comdat
	.weak	_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_ # -- Begin function _ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_,@function
_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_: # @_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movq	%r8, -32(%rbp)
	movq	%r9, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movslq	-20(%rbp), %rax
	movl	$8, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, %rcx
	movq	-64(%rbp), %rax                 # 8-byte Reload
	movq	%rcx, -56(%rbp)
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movl	-16(%rbp), %edx
	movl	-20(%rbp), %ecx
	movq	-32(%rbp), %r8
	movq	-40(%rbp), %r9
	movq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Type_get_contents@PLT
	movl	$0, -44(%rbp)
.LBB56_1:                               # =>This Inner Loop Header: Depth=1
	movl	-44(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.LBB56_4
# %bb.2:                                #   in Loop: Header=BB56_1 Depth=1
	movq	-56(%rbp), %rsi
	movslq	-44(%rbp), %rax
	shlq	$3, %rax
	addq	%rax, %rsi
	movq	16(%rbp), %rdi
	movslq	-44(%rbp), %rax
	shlq	$4, %rax
	addq	%rax, %rdi
	callq	_ZN3MPI8DatatypeaSERKP15ompi_datatype_t
# %bb.3:                                #   in Loop: Header=BB56_1 Depth=1
	movl	-44(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -44(%rbp)
	jmp	.LBB56_1
.LBB56_4:
	movq	-56(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB56_6
# %bb.5:
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB56_6:
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end56:
	.size	_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_, .Lfunc_end56-_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_,"axG",@progbits,_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_,comdat
	.weak	_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_ # -- Begin function _ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_,@function
_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_: # @_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-40(%rbp), %r8
	callq	MPI_Type_get_envelope@PLT
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end57:
	.size	_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_, .Lfunc_end57-_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Datatype8Get_nameEPcRi,"axG",@progbits,_ZNK3MPI8Datatype8Get_nameEPcRi,comdat
	.weak	_ZNK3MPI8Datatype8Get_nameEPcRi # -- Begin function _ZNK3MPI8Datatype8Get_nameEPcRi
	.p2align	4, 0x90
	.type	_ZNK3MPI8Datatype8Get_nameEPcRi,@function
_ZNK3MPI8Datatype8Get_nameEPcRi:        # @_ZNK3MPI8Datatype8Get_nameEPcRi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Type_get_name@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end58:
	.size	_ZNK3MPI8Datatype8Get_nameEPcRi, .Lfunc_end58-_ZNK3MPI8Datatype8Get_nameEPcRi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8Datatype8Set_attrEiPKv,"axG",@progbits,_ZN3MPI8Datatype8Set_attrEiPKv,comdat
	.weak	_ZN3MPI8Datatype8Set_attrEiPKv  # -- Begin function _ZN3MPI8Datatype8Set_attrEiPKv
	.p2align	4, 0x90
	.type	_ZN3MPI8Datatype8Set_attrEiPKv,@function
_ZN3MPI8Datatype8Set_attrEiPKv:         # @_ZN3MPI8Datatype8Set_attrEiPKv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	callq	MPI_Type_set_attr@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end59:
	.size	_ZN3MPI8Datatype8Set_attrEiPKv, .Lfunc_end59-_ZN3MPI8Datatype8Set_attrEiPKv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8Datatype8Set_nameEPKc,"axG",@progbits,_ZN3MPI8Datatype8Set_nameEPKc,comdat
	.weak	_ZN3MPI8Datatype8Set_nameEPKc   # -- Begin function _ZN3MPI8Datatype8Set_nameEPKc
	.p2align	4, 0x90
	.type	_ZN3MPI8Datatype8Set_nameEPKc,@function
_ZN3MPI8Datatype8Set_nameEPKc:          # @_ZN3MPI8Datatype8Set_nameEPKc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	callq	MPI_Type_set_name@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end60:
	.size	_ZN3MPI8Datatype8Set_nameEPKc, .Lfunc_end60-_ZN3MPI8Datatype8Set_nameEPKc
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6StatusD2Ev,"axG",@progbits,_ZN3MPI6StatusD2Ev,comdat
	.weak	_ZN3MPI6StatusD2Ev              # -- Begin function _ZN3MPI6StatusD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI6StatusD2Ev,@function
_ZN3MPI6StatusD2Ev:                     # @_ZN3MPI6StatusD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end61:
	.size	_ZN3MPI6StatusD2Ev, .Lfunc_end61-_ZN3MPI6StatusD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6StatusD0Ev,"axG",@progbits,_ZN3MPI6StatusD0Ev,comdat
	.weak	_ZN3MPI6StatusD0Ev              # -- Begin function _ZN3MPI6StatusD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI6StatusD0Ev,@function
_ZN3MPI6StatusD0Ev:                     # @_ZN3MPI6StatusD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI6StatusD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end62:
	.size	_ZN3MPI6StatusD0Ev, .Lfunc_end62-_ZN3MPI6StatusD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI6Status12Is_cancelledEv,"axG",@progbits,_ZNK3MPI6Status12Is_cancelledEv,comdat
	.weak	_ZNK3MPI6Status12Is_cancelledEv # -- Begin function _ZNK3MPI6Status12Is_cancelledEv
	.p2align	4, 0x90
	.type	_ZNK3MPI6Status12Is_cancelledEv,@function
_ZNK3MPI6Status12Is_cancelledEv:        # @_ZNK3MPI6Status12Is_cancelledEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Test_cancelled@PLT
	cmpl	$0, -12(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end63:
	.size	_ZNK3MPI6Status12Is_cancelledEv, .Lfunc_end63-_ZNK3MPI6Status12Is_cancelledEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE,"axG",@progbits,_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE,comdat
	.weak	_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE # -- Begin function _ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE
	.p2align	4, 0x90
	.type	_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE,@function
_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE: # @_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	addq	$8, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-32(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	leaq	-20(%rbp), %rdx
	callq	MPI_Get_elements@PLT
	movl	-20(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end64:
	.size	_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE, .Lfunc_end64-_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI6Status10Get_sourceEv,"axG",@progbits,_ZNK3MPI6Status10Get_sourceEv,comdat
	.weak	_ZNK3MPI6Status10Get_sourceEv   # -- Begin function _ZNK3MPI6Status10Get_sourceEv
	.p2align	4, 0x90
	.type	_ZNK3MPI6Status10Get_sourceEv,@function
_ZNK3MPI6Status10Get_sourceEv:          # @_ZNK3MPI6Status10Get_sourceEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	8(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end65:
	.size	_ZNK3MPI6Status10Get_sourceEv, .Lfunc_end65-_ZNK3MPI6Status10Get_sourceEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6Status10Set_sourceEi,"axG",@progbits,_ZN3MPI6Status10Set_sourceEi,comdat
	.weak	_ZN3MPI6Status10Set_sourceEi    # -- Begin function _ZN3MPI6Status10Set_sourceEi
	.p2align	4, 0x90
	.type	_ZN3MPI6Status10Set_sourceEi,@function
_ZN3MPI6Status10Set_sourceEi:           # @_ZN3MPI6Status10Set_sourceEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end66:
	.size	_ZN3MPI6Status10Set_sourceEi, .Lfunc_end66-_ZN3MPI6Status10Set_sourceEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI6Status7Get_tagEv,"axG",@progbits,_ZNK3MPI6Status7Get_tagEv,comdat
	.weak	_ZNK3MPI6Status7Get_tagEv       # -- Begin function _ZNK3MPI6Status7Get_tagEv
	.p2align	4, 0x90
	.type	_ZNK3MPI6Status7Get_tagEv,@function
_ZNK3MPI6Status7Get_tagEv:              # @_ZNK3MPI6Status7Get_tagEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	12(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end67:
	.size	_ZNK3MPI6Status7Get_tagEv, .Lfunc_end67-_ZNK3MPI6Status7Get_tagEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6Status7Set_tagEi,"axG",@progbits,_ZN3MPI6Status7Set_tagEi,comdat
	.weak	_ZN3MPI6Status7Set_tagEi        # -- Begin function _ZN3MPI6Status7Set_tagEi
	.p2align	4, 0x90
	.type	_ZN3MPI6Status7Set_tagEi,@function
_ZN3MPI6Status7Set_tagEi:               # @_ZN3MPI6Status7Set_tagEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, 12(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end68:
	.size	_ZN3MPI6Status7Set_tagEi, .Lfunc_end68-_ZN3MPI6Status7Set_tagEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI6Status9Get_errorEv,"axG",@progbits,_ZNK3MPI6Status9Get_errorEv,comdat
	.weak	_ZNK3MPI6Status9Get_errorEv     # -- Begin function _ZNK3MPI6Status9Get_errorEv
	.p2align	4, 0x90
	.type	_ZNK3MPI6Status9Get_errorEv,@function
_ZNK3MPI6Status9Get_errorEv:            # @_ZNK3MPI6Status9Get_errorEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	16(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end69:
	.size	_ZNK3MPI6Status9Get_errorEv, .Lfunc_end69-_ZNK3MPI6Status9Get_errorEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6Status9Set_errorEi,"axG",@progbits,_ZN3MPI6Status9Set_errorEi,comdat
	.weak	_ZN3MPI6Status9Set_errorEi      # -- Begin function _ZN3MPI6Status9Set_errorEi
	.p2align	4, 0x90
	.type	_ZN3MPI6Status9Set_errorEi,@function
_ZN3MPI6Status9Set_errorEi:             # @_ZN3MPI6Status9Set_errorEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, 16(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end70:
	.size	_ZN3MPI6Status9Set_errorEi, .Lfunc_end70-_ZN3MPI6Status9Set_errorEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi,"axG",@progbits,_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi,comdat
	.weak	_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi # -- Begin function _ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi
	.p2align	4, 0x90
	.type	_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi,@function
_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi: # @_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-8(%rbp), %rax
	addq	$8, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-32(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	movl	-20(%rbp), %edx
	callq	MPI_Status_set_elements@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end71:
	.size	_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi, .Lfunc_end71-_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6Status13Set_cancelledEb,"axG",@progbits,_ZN3MPI6Status13Set_cancelledEb,comdat
	.weak	_ZN3MPI6Status13Set_cancelledEb # -- Begin function _ZN3MPI6Status13Set_cancelledEb
	.p2align	4, 0x90
	.type	_ZN3MPI6Status13Set_cancelledEb,@function
_ZN3MPI6Status13Set_cancelledEb:        # @_ZN3MPI6Status13Set_cancelledEb
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movb	%sil, %al
	movq	%rdi, -8(%rbp)
	andb	$1, %al
	movb	%al, -9(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	movb	-9(%rbp), %al
	andb	$1, %al
	movzbl	%al, %esi
	callq	MPI_Status_set_cancelled@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end72:
	.size	_ZN3MPI6Status13Set_cancelledEb, .Lfunc_end72-_ZN3MPI6Status13Set_cancelledEb
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7RequestD2Ev,"axG",@progbits,_ZN3MPI7RequestD2Ev,comdat
	.weak	_ZN3MPI7RequestD2Ev             # -- Begin function _ZN3MPI7RequestD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI7RequestD2Ev,@function
_ZN3MPI7RequestD2Ev:                    # @_ZN3MPI7RequestD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end73:
	.size	_ZN3MPI7RequestD2Ev, .Lfunc_end73-_ZN3MPI7RequestD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7RequestD0Ev,"axG",@progbits,_ZN3MPI7RequestD0Ev,comdat
	.weak	_ZN3MPI7RequestD0Ev             # -- Begin function _ZN3MPI7RequestD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI7RequestD0Ev,@function
_ZN3MPI7RequestD0Ev:                    # @_ZN3MPI7RequestD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI7RequestD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end74:
	.size	_ZN3MPI7RequestD0Ev, .Lfunc_end74-_ZN3MPI7RequestD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7Request4WaitEv,"axG",@progbits,_ZN3MPI7Request4WaitEv,comdat
	.weak	_ZN3MPI7Request4WaitEv          # -- Begin function _ZN3MPI7Request4WaitEv
	.p2align	4, 0x90
	.type	_ZN3MPI7Request4WaitEv,@function
_ZN3MPI7Request4WaitEv:                 # @_ZN3MPI7Request4WaitEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	xorl	%eax, %eax
	movl	%eax, %esi
	callq	MPI_Wait@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end75:
	.size	_ZN3MPI7Request4WaitEv, .Lfunc_end75-_ZN3MPI7Request4WaitEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7Request4TestERNS_6StatusE,"axG",@progbits,_ZN3MPI7Request4TestERNS_6StatusE,comdat
	.weak	_ZN3MPI7Request4TestERNS_6StatusE # -- Begin function _ZN3MPI7Request4TestERNS_6StatusE
	.p2align	4, 0x90
	.type	_ZN3MPI7Request4TestERNS_6StatusE,@function
_ZN3MPI7Request4TestERNS_6StatusE:      # @_ZN3MPI7Request4TestERNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	movq	-16(%rbp), %rdx
	addq	$8, %rdx
	leaq	-20(%rbp), %rsi
	callq	MPI_Test@PLT
	cmpl	$0, -20(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end76:
	.size	_ZN3MPI7Request4TestERNS_6StatusE, .Lfunc_end76-_ZN3MPI7Request4TestERNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7Request4TestEv,"axG",@progbits,_ZN3MPI7Request4TestEv,comdat
	.weak	_ZN3MPI7Request4TestEv          # -- Begin function _ZN3MPI7Request4TestEv
	.p2align	4, 0x90
	.type	_ZN3MPI7Request4TestEv,@function
_ZN3MPI7Request4TestEv:                 # @_ZN3MPI7Request4TestEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	leaq	-12(%rbp), %rsi
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	MPI_Test@PLT
	cmpl	$0, -12(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end77:
	.size	_ZN3MPI7Request4TestEv, .Lfunc_end77-_ZN3MPI7Request4TestEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7Request4FreeEv,"axG",@progbits,_ZN3MPI7Request4FreeEv,comdat
	.weak	_ZN3MPI7Request4FreeEv          # -- Begin function _ZN3MPI7Request4FreeEv
	.p2align	4, 0x90
	.type	_ZN3MPI7Request4FreeEv,@function
_ZN3MPI7Request4FreeEv:                 # @_ZN3MPI7Request4FreeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Request_free@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end78:
	.size	_ZN3MPI7Request4FreeEv, .Lfunc_end78-_ZN3MPI7Request4FreeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI7Request6CancelEv,"axG",@progbits,_ZNK3MPI7Request6CancelEv,comdat
	.weak	_ZNK3MPI7Request6CancelEv       # -- Begin function _ZNK3MPI7Request6CancelEv
	.p2align	4, 0x90
	.type	_ZNK3MPI7Request6CancelEv,@function
_ZNK3MPI7Request6CancelEv:              # @_ZNK3MPI7Request6CancelEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Cancel@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end79:
	.size	_ZNK3MPI7Request6CancelEv, .Lfunc_end79-_ZNK3MPI7Request6CancelEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI7Request10Get_statusERNS_6StatusE,"axG",@progbits,_ZNK3MPI7Request10Get_statusERNS_6StatusE,comdat
	.weak	_ZNK3MPI7Request10Get_statusERNS_6StatusE # -- Begin function _ZNK3MPI7Request10Get_statusERNS_6StatusE
	.p2align	4, 0x90
	.type	_ZNK3MPI7Request10Get_statusERNS_6StatusE,@function
_ZNK3MPI7Request10Get_statusERNS_6StatusE: # @_ZNK3MPI7Request10Get_statusERNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movl	$0, -20(%rbp)
	movq	8(%rax), %rdi
	leaq	-20(%rbp), %rsi
	leaq	-48(%rbp), %rdx
	callq	MPI_Request_get_status@PLT
	cmpl	$0, -20(%rbp)
	je	.LBB80_2
# %bb.1:
	movq	-16(%rbp), %rdi
	leaq	-48(%rbp), %rsi
	callq	_ZN3MPI6StatusaSERK20ompi_status_public_t
.LBB80_2:
	cmpl	$0, -20(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end80:
	.size	_ZNK3MPI7Request10Get_statusERNS_6StatusE, .Lfunc_end80-_ZNK3MPI7Request10Get_statusERNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI7Request10Get_statusEv,"axG",@progbits,_ZNK3MPI7Request10Get_statusEv,comdat
	.weak	_ZNK3MPI7Request10Get_statusEv  # -- Begin function _ZNK3MPI7Request10Get_statusEv
	.p2align	4, 0x90
	.type	_ZNK3MPI7Request10Get_statusEv,@function
_ZNK3MPI7Request10Get_statusEv:         # @_ZNK3MPI7Request10Get_statusEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	MPI_Request_get_status@PLT
	cmpl	$0, -12(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end81:
	.size	_ZNK3MPI7Request10Get_statusEv, .Lfunc_end81-_ZNK3MPI7Request10Get_statusEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8PrequestD2Ev,"axG",@progbits,_ZN3MPI8PrequestD2Ev,comdat
	.weak	_ZN3MPI8PrequestD2Ev            # -- Begin function _ZN3MPI8PrequestD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8PrequestD2Ev,@function
_ZN3MPI8PrequestD2Ev:                   # @_ZN3MPI8PrequestD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI7RequestD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end82:
	.size	_ZN3MPI8PrequestD2Ev, .Lfunc_end82-_ZN3MPI8PrequestD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8PrequestD0Ev,"axG",@progbits,_ZN3MPI8PrequestD0Ev,comdat
	.weak	_ZN3MPI8PrequestD0Ev            # -- Begin function _ZN3MPI8PrequestD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8PrequestD0Ev,@function
_ZN3MPI8PrequestD0Ev:                   # @_ZN3MPI8PrequestD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI8PrequestD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end83:
	.size	_ZN3MPI8PrequestD0Ev, .Lfunc_end83-_ZN3MPI8PrequestD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8GrequestD2Ev,"axG",@progbits,_ZN3MPI8GrequestD2Ev,comdat
	.weak	_ZN3MPI8GrequestD2Ev            # -- Begin function _ZN3MPI8GrequestD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8GrequestD2Ev,@function
_ZN3MPI8GrequestD2Ev:                   # @_ZN3MPI8GrequestD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI7RequestD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end84:
	.size	_ZN3MPI8GrequestD2Ev, .Lfunc_end84-_ZN3MPI8GrequestD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8GrequestD0Ev,"axG",@progbits,_ZN3MPI8GrequestD0Ev,comdat
	.weak	_ZN3MPI8GrequestD0Ev            # -- Begin function _ZN3MPI8GrequestD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8GrequestD0Ev,@function
_ZN3MPI8GrequestD0Ev:                   # @_ZN3MPI8GrequestD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI8GrequestD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end85:
	.size	_ZN3MPI8GrequestD0Ev, .Lfunc_end85-_ZN3MPI8GrequestD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI5GroupD2Ev,"axG",@progbits,_ZN3MPI5GroupD2Ev,comdat
	.weak	_ZN3MPI5GroupD2Ev               # -- Begin function _ZN3MPI5GroupD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI5GroupD2Ev,@function
_ZN3MPI5GroupD2Ev:                      # @_ZN3MPI5GroupD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end86:
	.size	_ZN3MPI5GroupD2Ev, .Lfunc_end86-_ZN3MPI5GroupD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI5GroupD0Ev,"axG",@progbits,_ZN3MPI5GroupD0Ev,comdat
	.weak	_ZN3MPI5GroupD0Ev               # -- Begin function _ZN3MPI5GroupD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI5GroupD0Ev,@function
_ZN3MPI5GroupD0Ev:                      # @_ZN3MPI5GroupD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI5GroupD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end87:
	.size	_ZN3MPI5GroupD0Ev, .Lfunc_end87-_ZN3MPI5GroupD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5Group8Get_rankEv,"axG",@progbits,_ZNK3MPI5Group8Get_rankEv,comdat
	.weak	_ZNK3MPI5Group8Get_rankEv       # -- Begin function _ZNK3MPI5Group8Get_rankEv
	.p2align	4, 0x90
	.type	_ZNK3MPI5Group8Get_rankEv,@function
_ZNK3MPI5Group8Get_rankEv:              # @_ZNK3MPI5Group8Get_rankEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Group_rank@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end88:
	.size	_ZNK3MPI5Group8Get_rankEv, .Lfunc_end88-_ZNK3MPI5Group8Get_rankEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5Group4InclEiPKi,"axG",@progbits,_ZNK3MPI5Group4InclEiPKi,comdat
	.weak	_ZNK3MPI5Group4InclEiPKi        # -- Begin function _ZNK3MPI5Group4InclEiPKi
	.p2align	4, 0x90
	.type	_ZNK3MPI5Group4InclEiPKi,@function
_ZNK3MPI5Group4InclEiPKi:               # @_ZNK3MPI5Group4InclEiPKi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rcx
	callq	MPI_Group_incl@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end89:
	.size	_ZNK3MPI5Group4InclEiPKi, .Lfunc_end89-_ZNK3MPI5Group4InclEiPKi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5Group4ExclEiPKi,"axG",@progbits,_ZNK3MPI5Group4ExclEiPKi,comdat
	.weak	_ZNK3MPI5Group4ExclEiPKi        # -- Begin function _ZNK3MPI5Group4ExclEiPKi
	.p2align	4, 0x90
	.type	_ZNK3MPI5Group4ExclEiPKi,@function
_ZNK3MPI5Group4ExclEiPKi:               # @_ZNK3MPI5Group4ExclEiPKi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rcx
	callq	MPI_Group_excl@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end90:
	.size	_ZNK3MPI5Group4ExclEiPKi, .Lfunc_end90-_ZNK3MPI5Group4ExclEiPKi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5Group10Range_inclEiPA3_Ki,"axG",@progbits,_ZNK3MPI5Group10Range_inclEiPA3_Ki,comdat
	.weak	_ZNK3MPI5Group10Range_inclEiPA3_Ki # -- Begin function _ZNK3MPI5Group10Range_inclEiPA3_Ki
	.p2align	4, 0x90
	.type	_ZNK3MPI5Group10Range_inclEiPA3_Ki,@function
_ZNK3MPI5Group10Range_inclEiPA3_Ki:     # @_ZNK3MPI5Group10Range_inclEiPA3_Ki
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rcx
	callq	MPI_Group_range_incl@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end91:
	.size	_ZNK3MPI5Group10Range_inclEiPA3_Ki, .Lfunc_end91-_ZNK3MPI5Group10Range_inclEiPA3_Ki
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5Group10Range_exclEiPA3_Ki,"axG",@progbits,_ZNK3MPI5Group10Range_exclEiPA3_Ki,comdat
	.weak	_ZNK3MPI5Group10Range_exclEiPA3_Ki # -- Begin function _ZNK3MPI5Group10Range_exclEiPA3_Ki
	.p2align	4, 0x90
	.type	_ZNK3MPI5Group10Range_exclEiPA3_Ki,@function
_ZNK3MPI5Group10Range_exclEiPA3_Ki:     # @_ZNK3MPI5Group10Range_exclEiPA3_Ki
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rcx
	callq	MPI_Group_range_excl@PLT
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	-40(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end92:
	.size	_ZNK3MPI5Group10Range_exclEiPA3_Ki, .Lfunc_end92-_ZNK3MPI5Group10Range_exclEiPA3_Ki
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI5Group4FreeEv,"axG",@progbits,_ZN3MPI5Group4FreeEv,comdat
	.weak	_ZN3MPI5Group4FreeEv            # -- Begin function _ZN3MPI5Group4FreeEv
	.p2align	4, 0x90
	.type	_ZN3MPI5Group4FreeEv,@function
_ZN3MPI5Group4FreeEv:                   # @_ZN3MPI5Group4FreeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Group_free@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end93:
	.size	_ZN3MPI5Group4FreeEv, .Lfunc_end93-_ZN3MPI5Group4FreeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4CommD2Ev,"axG",@progbits,_ZN3MPI4CommD2Ev,comdat
	.weak	_ZN3MPI4CommD2Ev                # -- Begin function _ZN3MPI4CommD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI4CommD2Ev,@function
_ZN3MPI4CommD2Ev:                       # @_ZN3MPI4CommD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI9Comm_NullD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end94:
	.size	_ZN3MPI4CommD2Ev, .Lfunc_end94-_ZN3MPI4CommD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4CommD0Ev,"axG",@progbits,_ZN3MPI4CommD0Ev,comdat
	.weak	_ZN3MPI4CommD0Ev                # -- Begin function _ZN3MPI4CommD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI4CommD0Ev,@function
_ZN3MPI4CommD0Ev:                       # @_ZN3MPI4CommD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	ud2
.Lfunc_end95:
	.size	_ZN3MPI4CommD0Ev, .Lfunc_end95-_ZN3MPI4CommD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE,"axG",@progbits,_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE,comdat
	.weak	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE # -- Begin function _ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE,@function
_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE: # @_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movq	8(%rax), %r9
	movq	16(%rbp), %rax
	addq	$8, %rax
	movq	%rax, (%rsp)
	callq	MPI_Recv@PLT
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end96:
	.size	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE, .Lfunc_end96-_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movq	8(%rax), %r9
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	movq	$0, (%rsp)
	callq	MPI_Recv@PLT
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end97:
	.size	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii, .Lfunc_end97-_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movq	8(%rax), %r9
	callq	MPI_Bsend@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end98:
	.size	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii, .Lfunc_end98-_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movq	8(%rax), %r9
	callq	MPI_Ssend@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end99:
	.size	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii, .Lfunc_end99-_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movq	8(%rax), %r9
	callq	MPI_Rsend@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end100:
	.size	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii, .Lfunc_end100-_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Isend@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI7RequestC2EP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end101:
	.size	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii, .Lfunc_end101-_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Ibsend@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI7RequestC2EP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end102:
	.size	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii, .Lfunc_end102-_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Issend@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI7RequestC2EP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end103:
	.size	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii, .Lfunc_end103-_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Irsend@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI7RequestC2EP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end104:
	.size	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii, .Lfunc_end104-_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Irecv@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI7RequestC2EP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end105:
	.size	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii, .Lfunc_end105-_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6IprobeEiiRNS_6StatusE,"axG",@progbits,_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE,comdat
	.weak	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE # -- Begin function _ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE,@function
_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE:    # @_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movl	-16(%rbp), %esi
	movq	8(%rax), %rdx
	movq	-24(%rbp), %r8
	addq	$8, %r8
	leaq	-28(%rbp), %rcx
	callq	MPI_Iprobe@PLT
	cmpl	$0, -28(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end106:
	.size	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE, .Lfunc_end106-_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6IprobeEii,"axG",@progbits,_ZNK3MPI4Comm6IprobeEii,comdat
	.weak	_ZNK3MPI4Comm6IprobeEii         # -- Begin function _ZNK3MPI4Comm6IprobeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6IprobeEii,@function
_ZNK3MPI4Comm6IprobeEii:                # @_ZNK3MPI4Comm6IprobeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movl	-16(%rbp), %esi
	movq	8(%rax), %rdx
	leaq	-20(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %r8d
	callq	MPI_Iprobe@PLT
	cmpl	$0, -20(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end107:
	.size	_ZNK3MPI4Comm6IprobeEii, .Lfunc_end107-_ZNK3MPI4Comm6IprobeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5ProbeEiiRNS_6StatusE,"axG",@progbits,_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE,comdat
	.weak	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE # -- Begin function _ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE,@function
_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE:     # @_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movl	-16(%rbp), %esi
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rcx
	addq	$8, %rcx
	callq	MPI_Probe@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end108:
	.size	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE, .Lfunc_end108-_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5ProbeEii,"axG",@progbits,_ZNK3MPI4Comm5ProbeEii,comdat
	.weak	_ZNK3MPI4Comm5ProbeEii          # -- Begin function _ZNK3MPI4Comm5ProbeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5ProbeEii,@function
_ZNK3MPI4Comm5ProbeEii:                 # @_ZNK3MPI4Comm5ProbeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movl	-16(%rbp), %esi
	movq	8(%rax), %rdx
	xorl	%eax, %eax
	movl	%eax, %ecx
	callq	MPI_Probe@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end109:
	.size	_ZNK3MPI4Comm5ProbeEii, .Lfunc_end109-_ZNK3MPI4Comm5ProbeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Send_init@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI8PrequestC2ERKP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end110:
	.size	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii, .Lfunc_end110-_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Bsend_init@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI8PrequestC2ERKP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end111:
	.size	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii, .Lfunc_end111-_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Ssend_init@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI8PrequestC2ERKP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end112:
	.size	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii, .Lfunc_end112-_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Rsend_init@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI8PrequestC2ERKP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end113:
	.size	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii, .Lfunc_end113-_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii,"axG",@progbits,_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii,comdat
	.weak	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii # -- Begin function _ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii,@function
_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii: # @_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	movq	%rax, (%rsp)
	callq	MPI_Recv_init@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI8PrequestC2ERKP14ompi_request_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end114:
	.size	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii, .Lfunc_end114-_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE,"axG",@progbits,_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE,comdat
	.weak	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE # -- Begin function _ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE,@function
_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE: # @_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	subq	$136, %rsp
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	56(%rbp), %rax
	movl	48(%rbp), %eax
	movl	40(%rbp), %eax
	movq	32(%rbp), %rax
	movl	24(%rbp), %eax
	movq	16(%rbp), %rax
	movq	%rdi, -32(%rbp)
	movq	%rsi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movq	%rcx, -56(%rbp)
	movl	%r8d, -60(%rbp)
	movl	%r9d, -64(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -112(%rbp)                # 8-byte Spill
	movl	-44(%rbp), %eax
	movl	%eax, -100(%rbp)                # 4-byte Spill
	movq	-56(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-60(%rbp), %eax
	movl	%eax, -88(%rbp)                 # 4-byte Spill
	movl	-64(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	24(%rbp), %r15d
	movq	32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-112(%rbp), %rdi                # 8-byte Reload
	movl	-100(%rbp), %esi                # 4-byte Reload
	movq	-96(%rbp), %rdx                 # 8-byte Reload
	movl	-88(%rbp), %ecx                 # 4-byte Reload
	movl	-84(%rbp), %r8d                 # 4-byte Reload
	movq	-80(%rbp), %r9                  # 8-byte Reload
	movq	%rax, %r14
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movl	40(%rbp), %ebx
	movl	48(%rbp), %r11d
	movq	8(%rax), %r10
	movq	56(%rbp), %rax
	addq	$8, %rax
	movl	%r15d, (%rsp)
	movq	%r14, 8(%rsp)
	movl	%ebx, 16(%rsp)
	movl	%r11d, 24(%rsp)
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	MPI_Sendrecv@PLT
	addq	$136, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end115:
	.size	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE, .Lfunc_end115-_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii,"axG",@progbits,_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii,comdat
	.weak	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii # -- Begin function _ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii,@function
_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii: # @_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	subq	$136, %rsp
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movl	48(%rbp), %eax
	movl	40(%rbp), %eax
	movq	32(%rbp), %rax
	movl	24(%rbp), %eax
	movq	16(%rbp), %rax
	movq	%rdi, -32(%rbp)
	movq	%rsi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movq	%rcx, -56(%rbp)
	movl	%r8d, -60(%rbp)
	movl	%r9d, -64(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -112(%rbp)                # 8-byte Spill
	movl	-44(%rbp), %eax
	movl	%eax, -100(%rbp)                # 4-byte Spill
	movq	-56(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-60(%rbp), %eax
	movl	%eax, -88(%rbp)                 # 4-byte Spill
	movl	-64(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	24(%rbp), %r14d
	movq	32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-112(%rbp), %rdi                # 8-byte Reload
	movl	-100(%rbp), %esi                # 4-byte Reload
	movq	-96(%rbp), %rdx                 # 8-byte Reload
	movl	-88(%rbp), %ecx                 # 4-byte Reload
	movl	-84(%rbp), %r8d                 # 4-byte Reload
	movq	-80(%rbp), %r9                  # 8-byte Reload
	movq	%rax, %rbx
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movl	40(%rbp), %r11d
	movl	48(%rbp), %r10d
	movq	8(%rax), %rax
	xorl	%r15d, %r15d
                                        # kill: def $r15 killed $r15d
	movl	%r14d, (%rsp)
	movq	%rbx, 8(%rsp)
	movl	%r11d, 16(%rsp)
	movl	%r10d, 24(%rsp)
	movq	%rax, 32(%rsp)
	movq	$0, 40(%rsp)
	callq	MPI_Sendrecv@PLT
	addq	$136, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end116:
	.size	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii, .Lfunc_end116-_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE,"axG",@progbits,_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE,comdat
	.weak	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE # -- Begin function _ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE,@function
_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE: # @_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	32(%rbp), %rax
	movl	24(%rbp), %eax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movl	16(%rbp), %r9d
	movl	24(%rbp), %r11d
	movq	8(%rax), %r10
	movq	32(%rbp), %rax
	addq	$8, %rax
	movl	%r11d, (%rsp)
	movq	%r10, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Sendrecv_replace@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end117:
	.size	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE, .Lfunc_end117-_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii,"axG",@progbits,_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii,comdat
	.weak	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii # -- Begin function _ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii,@function
_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii: # @_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movl	24(%rbp), %eax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	movl	16(%rbp), %r9d
	movl	24(%rbp), %r10d
	movq	8(%rax), %rax
	xorl	%r11d, %r11d
                                        # kill: def $r11 killed $r11d
	movl	%r10d, (%rsp)
	movq	%rax, 8(%rsp)
	movq	$0, 16(%rsp)
	callq	MPI_Sendrecv_replace@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end118:
	.size	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii, .Lfunc_end118-_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9Get_groupEv,"axG",@progbits,_ZNK3MPI4Comm9Get_groupEv,comdat
	.weak	_ZNK3MPI4Comm9Get_groupEv       # -- Begin function _ZNK3MPI4Comm9Get_groupEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9Get_groupEv,@function
_ZNK3MPI4Comm9Get_groupEv:              # @_ZNK3MPI4Comm9Get_groupEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Comm_group@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end119:
	.size	_ZNK3MPI4Comm9Get_groupEv, .Lfunc_end119-_ZNK3MPI4Comm9Get_groupEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8Get_sizeEv,"axG",@progbits,_ZNK3MPI4Comm8Get_sizeEv,comdat
	.weak	_ZNK3MPI4Comm8Get_sizeEv        # -- Begin function _ZNK3MPI4Comm8Get_sizeEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8Get_sizeEv,@function
_ZNK3MPI4Comm8Get_sizeEv:               # @_ZNK3MPI4Comm8Get_sizeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Comm_size@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end120:
	.size	_ZNK3MPI4Comm8Get_sizeEv, .Lfunc_end120-_ZNK3MPI4Comm8Get_sizeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8Get_rankEv,"axG",@progbits,_ZNK3MPI4Comm8Get_rankEv,comdat
	.weak	_ZNK3MPI4Comm8Get_rankEv        # -- Begin function _ZNK3MPI4Comm8Get_rankEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8Get_rankEv,@function
_ZNK3MPI4Comm8Get_rankEv:               # @_ZNK3MPI4Comm8Get_rankEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Comm_rank@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end121:
	.size	_ZNK3MPI4Comm8Get_rankEv, .Lfunc_end121-_ZNK3MPI4Comm8Get_rankEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Comm4FreeEv,"axG",@progbits,_ZN3MPI4Comm4FreeEv,comdat
	.weak	_ZN3MPI4Comm4FreeEv             # -- Begin function _ZN3MPI4Comm4FreeEv
	.p2align	4, 0x90
	.type	_ZN3MPI4Comm4FreeEv,@function
_ZN3MPI4Comm4FreeEv:                    # @_ZN3MPI4Comm4FreeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Comm_free@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end122:
	.size	_ZN3MPI4Comm4FreeEv, .Lfunc_end122-_ZN3MPI4Comm4FreeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8Is_interEv,"axG",@progbits,_ZNK3MPI4Comm8Is_interEv,comdat
	.weak	_ZNK3MPI4Comm8Is_interEv        # -- Begin function _ZNK3MPI4Comm8Is_interEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8Is_interEv,@function
_ZNK3MPI4Comm8Is_interEv:               # @_ZNK3MPI4Comm8Is_interEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Comm_test_inter@PLT
	cmpl	$0, -12(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end123:
	.size	_ZNK3MPI4Comm8Is_interEv, .Lfunc_end123-_ZNK3MPI4Comm8Is_interEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm7BarrierEv,"axG",@progbits,_ZNK3MPI4Comm7BarrierEv,comdat
	.weak	_ZNK3MPI4Comm7BarrierEv         # -- Begin function _ZNK3MPI4Comm7BarrierEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm7BarrierEv,@function
_ZNK3MPI4Comm7BarrierEv:                # @_ZNK3MPI4Comm7BarrierEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	callq	MPI_Barrier@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end124:
	.size	_ZNK3MPI4Comm7BarrierEv, .Lfunc_end124-_ZNK3MPI4Comm7BarrierEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi,"axG",@progbits,_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi,comdat
	.weak	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi # -- Begin function _ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi,@function
_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi: # @_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movl	-52(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %ecx
	movq	8(%rax), %r8
	callq	MPI_Bcast@PLT
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end125:
	.size	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi, .Lfunc_end125-_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i,"axG",@progbits,_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i,comdat
	.weak	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i # -- Begin function _ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i,@function
_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i: # @_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movl	24(%rbp), %eax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	-44(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movl	-60(%rbp), %r8d                 # 4-byte Reload
	movq	%rax, %r9
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movl	24(%rbp), %r10d
	movq	8(%rax), %rax
	movl	%r10d, (%rsp)
	movq	%rax, 8(%rsp)
	callq	MPI_Gather@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end126:
	.size	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i, .Lfunc_end126-_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i,"axG",@progbits,_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i,comdat
	.weak	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i # -- Begin function _ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i,@function
_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i: # @_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movl	32(%rbp), %eax
	movq	24(%rbp), %rax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -92(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	24(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movl	-92(%rbp), %esi                 # 4-byte Reload
	movq	-88(%rbp), %rdx                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-72(%rbp), %r8                  # 8-byte Reload
	movq	-64(%rbp), %r9                  # 8-byte Reload
	movq	%rax, %r11
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movl	32(%rbp), %r10d
	movq	8(%rax), %rax
	movq	%r11, (%rsp)
	movl	%r10d, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Gatherv@PLT
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end127:
	.size	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i, .Lfunc_end127-_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i,"axG",@progbits,_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i,comdat
	.weak	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i # -- Begin function _ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i,@function
_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i: # @_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movl	24(%rbp), %eax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	-44(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movl	-60(%rbp), %r8d                 # 4-byte Reload
	movq	%rax, %r9
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movl	24(%rbp), %r10d
	movq	8(%rax), %rax
	movl	%r10d, (%rsp)
	movq	%rax, 8(%rsp)
	callq	MPI_Scatter@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end128:
	.size	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i, .Lfunc_end128-_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i,"axG",@progbits,_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i,comdat
	.weak	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i # -- Begin function _ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i,@function
_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i: # @_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movl	32(%rbp), %eax
	movq	24(%rbp), %rax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movq	-32(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	24(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movq	-96(%rbp), %rsi                 # 8-byte Reload
	movq	-88(%rbp), %rdx                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-72(%rbp), %r8                  # 8-byte Reload
	movl	-60(%rbp), %r9d                 # 4-byte Reload
	movq	%rax, %r11
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movl	32(%rbp), %r10d
	movq	8(%rax), %rax
	movq	%r11, (%rsp)
	movl	%r10d, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Scatterv@PLT
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end129:
	.size	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i, .Lfunc_end129-_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_,"axG",@progbits,_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_,comdat
	.weak	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_ # -- Begin function _ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_,@function
_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_: # @_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	-44(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movl	-60(%rbp), %r8d                 # 4-byte Reload
	movq	%rax, %r9
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%rax, (%rsp)
	callq	MPI_Allgather@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end130:
	.size	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_, .Lfunc_end130-_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_,"axG",@progbits,_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_,comdat
	.weak	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_ # -- Begin function _ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_,@function
_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_: # @_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	24(%rbp), %rax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -92(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	24(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movl	-92(%rbp), %esi                 # 4-byte Reload
	movq	-88(%rbp), %rdx                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-72(%rbp), %r8                  # 8-byte Reload
	movq	-64(%rbp), %r9                  # 8-byte Reload
	movq	%rax, %r10
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	MPI_Allgatherv@PLT
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end131:
	.size	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_, .Lfunc_end131-_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_,"axG",@progbits,_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_,comdat
	.weak	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_ # -- Begin function _ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_,@function
_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_: # @_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movl	%r9d, -44(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	-44(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	16(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movl	-84(%rbp), %esi                 # 4-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movl	-60(%rbp), %r8d                 # 4-byte Reload
	movq	%rax, %r9
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%rax, (%rsp)
	callq	MPI_Alltoall@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end132:
	.size	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_, .Lfunc_end132-_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_,"axG",@progbits,_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_,comdat
	.weak	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_ # -- Begin function _ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_,@function
_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_: # @_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$144, %rsp
	movq	32(%rbp), %rax
	movq	24(%rbp), %rax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -112(%rbp)                # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movq	-32(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	16(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	24(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-112(%rbp), %rdi                # 8-byte Reload
	movq	-104(%rbp), %rsi                # 8-byte Reload
	movq	-96(%rbp), %rdx                 # 8-byte Reload
	movq	-88(%rbp), %rcx                 # 8-byte Reload
	movq	-80(%rbp), %r8                  # 8-byte Reload
	movq	-72(%rbp), %r9                  # 8-byte Reload
	movq	-64(%rbp), %r11                 # 8-byte Reload
	movq	%rax, %r10
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%r11, (%rsp)
	movq	%r10, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Alltoallv@PLT
	addq	$144, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end133:
	.size	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_, .Lfunc_end133-_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_,"axG",@progbits,_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_,comdat
	.weak	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_ # -- Begin function _ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_,@function
_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_: # @_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%rbx
	subq	$120, %rsp
	.cfi_offset %rbx, -24
	movq	32(%rbp), %rax
	movq	24(%rbp), %rax
	movq	16(%rbp), %rax
	movq	%rdi, -16(%rbp)
	movq	%rsi, -24(%rbp)
	movq	%rdx, -32(%rbp)
	movq	%rcx, -40(%rbp)
	movq	%r8, -48(%rbp)
	movq	%r9, -56(%rbp)
	movq	-16(%rbp), %rdi
	movq	%rdi, -88(%rbp)                 # 8-byte Spill
	movq	(%rdi), %rax
	callq	*216(%rax)
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	shll	$1, %eax
	cltq
	movl	$8, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, -72(%rbp)
	movl	$0, -76(%rbp)
.LBB134_1:                              # =>This Inner Loop Header: Depth=1
	movl	-76(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.LBB134_4
# %bb.2:                                #   in Loop: Header=BB134_1 Depth=1
	movq	-48(%rbp), %rdi
	movslq	-76(%rbp), %rax
	shlq	$4, %rax
	addq	%rax, %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	movslq	-76(%rbp), %rcx
	movq	%rdx, (%rax,%rcx,8)
	movq	32(%rbp), %rdi
	movslq	-76(%rbp), %rax
	shlq	$4, %rax
	addq	%rax, %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	movl	-76(%rbp), %ecx
	addl	-60(%rbp), %ecx
	movslq	%ecx, %rcx
	movq	%rdx, (%rax,%rcx,8)
# %bb.3:                                #   in Loop: Header=BB134_1 Depth=1
	movl	-76(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -76(%rbp)
	jmp	.LBB134_1
.LBB134_4:
	movq	-88(%rbp), %rax                 # 8-byte Reload
	movq	-24(%rbp), %rdi
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rdx
	movq	-72(%rbp), %rcx
	movq	-56(%rbp), %r8
	movq	16(%rbp), %r9
	movq	24(%rbp), %r11
	movq	-72(%rbp), %r10
	movslq	-60(%rbp), %rbx
	shlq	$3, %rbx
	addq	%rbx, %r10
	movq	8(%rax), %rax
	movq	%r11, (%rsp)
	movq	%r10, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Alltoallw@PLT
	movq	-72(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB134_6
# %bb.5:
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB134_6:
	addq	$120, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end134:
	.size	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_, .Lfunc_end134-_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi,"axG",@progbits,_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi,comdat
	.weak	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi # -- Begin function _ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi,@function
_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi: # @_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI2OpcvP9ompi_op_tEv
	movq	-88(%rbp), %rdi                 # 8-byte Reload
	movq	-80(%rbp), %rsi                 # 8-byte Reload
	movl	-68(%rbp), %edx                 # 4-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %r8
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movl	16(%rbp), %r9d
	movq	8(%rax), %rax
	movq	%rax, (%rsp)
	callq	MPI_Reduce@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end135:
	.size	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi, .Lfunc_end135-_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE,"axG",@progbits,_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE,comdat
	.weak	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE # -- Begin function _ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE,@function
_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE: # @_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI2OpcvP9ompi_op_tEv
	movq	-88(%rbp), %rdi                 # 8-byte Reload
	movq	-80(%rbp), %rsi                 # 8-byte Reload
	movl	-68(%rbp), %edx                 # 4-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %r8
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %r9
	callq	MPI_Allreduce@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end136:
	.size	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE, .Lfunc_end136-_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE,"axG",@progbits,_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE,comdat
	.weak	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE # -- Begin function _ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE,@function
_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE: # @_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI2OpcvP9ompi_op_tEv
	movq	-88(%rbp), %rdi                 # 8-byte Reload
	movq	-80(%rbp), %rsi                 # 8-byte Reload
	movq	-72(%rbp), %rdx                 # 8-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %r8
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %r9
	callq	MPI_Reduce_scatter@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end137:
	.size	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE, .Lfunc_end137-_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Comm10DisconnectEv,"axG",@progbits,_ZN3MPI4Comm10DisconnectEv,comdat
	.weak	_ZN3MPI4Comm10DisconnectEv      # -- Begin function _ZN3MPI4Comm10DisconnectEv
	.p2align	4, 0x90
	.type	_ZN3MPI4Comm10DisconnectEv,@function
_ZN3MPI4Comm10DisconnectEv:             # @_ZN3MPI4Comm10DisconnectEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Comm_disconnect@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end138:
	.size	_ZN3MPI4Comm10DisconnectEv, .Lfunc_end138-_ZN3MPI4Comm10DisconnectEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8Get_nameEPcRi,"axG",@progbits,_ZNK3MPI4Comm8Get_nameEPcRi,comdat
	.weak	_ZNK3MPI4Comm8Get_nameEPcRi     # -- Begin function _ZNK3MPI4Comm8Get_nameEPcRi
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8Get_nameEPcRi,@function
_ZNK3MPI4Comm8Get_nameEPcRi:            # @_ZNK3MPI4Comm8Get_nameEPcRi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Comm_get_name@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end139:
	.size	_ZNK3MPI4Comm8Get_nameEPcRi, .Lfunc_end139-_ZNK3MPI4Comm8Get_nameEPcRi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Comm8Set_nameEPKc,"axG",@progbits,_ZN3MPI4Comm8Set_nameEPKc,comdat
	.weak	_ZN3MPI4Comm8Set_nameEPKc       # -- Begin function _ZN3MPI4Comm8Set_nameEPKc
	.p2align	4, 0x90
	.type	_ZN3MPI4Comm8Set_nameEPKc,@function
_ZN3MPI4Comm8Set_nameEPKc:              # @_ZN3MPI4Comm8Set_nameEPKc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	callq	MPI_Comm_set_name@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end140:
	.size	_ZN3MPI4Comm8Set_nameEPKc, .Lfunc_end140-_ZN3MPI4Comm8Set_nameEPKc
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm12Get_topologyEv,"axG",@progbits,_ZNK3MPI4Comm12Get_topologyEv,comdat
	.weak	_ZNK3MPI4Comm12Get_topologyEv   # -- Begin function _ZNK3MPI4Comm12Get_topologyEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm12Get_topologyEv,@function
_ZNK3MPI4Comm12Get_topologyEv:          # @_ZNK3MPI4Comm12Get_topologyEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Topo_test@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end141:
	.size	_ZNK3MPI4Comm12Get_topologyEv, .Lfunc_end141-_ZNK3MPI4Comm12Get_topologyEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Comm5AbortEi,"axG",@progbits,_ZN3MPI4Comm5AbortEi,comdat
	.weak	_ZN3MPI4Comm5AbortEi            # -- Begin function _ZN3MPI4Comm5AbortEi
	.p2align	4, 0x90
	.type	_ZN3MPI4Comm5AbortEi,@function
_ZN3MPI4Comm5AbortEi:                   # @_ZN3MPI4Comm5AbortEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	callq	MPI_Abort@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end142:
	.size	_ZN3MPI4Comm5AbortEi, .Lfunc_end142-_ZN3MPI4Comm5AbortEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE,"axG",@progbits,_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE,comdat
	.weak	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE # -- Begin function _ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.p2align	4, 0x90
	.type	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE,@function
_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE: # @_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv
	movq	-24(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	callq	MPI_Comm_set_errhandler@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end143:
	.size	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE, .Lfunc_end143-_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm14Get_errhandlerEv,"axG",@progbits,_ZNK3MPI4Comm14Get_errhandlerEv,comdat
	.weak	_ZNK3MPI4Comm14Get_errhandlerEv # -- Begin function _ZNK3MPI4Comm14Get_errhandlerEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm14Get_errhandlerEv,@function
_ZNK3MPI4Comm14Get_errhandlerEv:        # @_ZNK3MPI4Comm14Get_errhandlerEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Comm_get_errhandler@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end144:
	.size	_ZNK3MPI4Comm14Get_errhandlerEv, .Lfunc_end144-_ZNK3MPI4Comm14Get_errhandlerEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8Set_attrEiPKv,"axG",@progbits,_ZNK3MPI4Comm8Set_attrEiPKv,comdat
	.weak	_ZNK3MPI4Comm8Set_attrEiPKv     # -- Begin function _ZNK3MPI4Comm8Set_attrEiPKv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8Set_attrEiPKv,@function
_ZNK3MPI4Comm8Set_attrEiPKv:            # @_ZNK3MPI4Comm8Set_attrEiPKv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	callq	MPI_Comm_set_attr@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end145:
	.size	_ZNK3MPI4Comm8Set_attrEiPKv, .Lfunc_end145-_ZNK3MPI4Comm8Set_attrEiPKv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Comm8Get_attrEiPv,"axG",@progbits,_ZNK3MPI4Comm8Get_attrEiPv,comdat
	.weak	_ZNK3MPI4Comm8Get_attrEiPv      # -- Begin function _ZNK3MPI4Comm8Get_attrEiPv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Comm8Get_attrEiPv,@function
_ZNK3MPI4Comm8Get_attrEiPv:             # @_ZNK3MPI4Comm8Get_attrEiPv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rcx
	callq	MPI_Comm_get_attr@PLT
	cmpl	$0, -28(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end146:
	.size	_ZNK3MPI4Comm8Get_attrEiPv, .Lfunc_end146-_ZNK3MPI4Comm8Get_attrEiPv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Comm11Delete_attrEi,"axG",@progbits,_ZN3MPI4Comm11Delete_attrEi,comdat
	.weak	_ZN3MPI4Comm11Delete_attrEi     # -- Begin function _ZN3MPI4Comm11Delete_attrEi
	.p2align	4, 0x90
	.type	_ZN3MPI4Comm11Delete_attrEi,@function
_ZN3MPI4Comm11Delete_attrEi:            # @_ZN3MPI4Comm11Delete_attrEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	callq	MPI_Comm_delete_attr@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end147:
	.size	_ZN3MPI4Comm11Delete_attrEi, .Lfunc_end147-_ZN3MPI4Comm11Delete_attrEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI3WinD2Ev,"axG",@progbits,_ZN3MPI3WinD2Ev,comdat
	.weak	_ZN3MPI3WinD2Ev                 # -- Begin function _ZN3MPI3WinD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI3WinD2Ev,@function
_ZN3MPI3WinD2Ev:                        # @_ZN3MPI3WinD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end148:
	.size	_ZN3MPI3WinD2Ev, .Lfunc_end148-_ZN3MPI3WinD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI3WinD0Ev,"axG",@progbits,_ZN3MPI3WinD0Ev,comdat
	.weak	_ZN3MPI3WinD0Ev                 # -- Begin function _ZN3MPI3WinD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI3WinD0Ev,@function
_ZN3MPI3WinD0Ev:                        # @_ZN3MPI3WinD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI3WinD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end149:
	.size	_ZN3MPI3WinD0Ev, .Lfunc_end149-_ZN3MPI3WinD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win14Get_errhandlerEv,"axG",@progbits,_ZNK3MPI3Win14Get_errhandlerEv,comdat
	.weak	_ZNK3MPI3Win14Get_errhandlerEv  # -- Begin function _ZNK3MPI3Win14Get_errhandlerEv
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win14Get_errhandlerEv,@function
_ZNK3MPI3Win14Get_errhandlerEv:         # @_ZNK3MPI3Win14Get_errhandlerEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Win_get_errhandler@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end150:
	.size	_ZNK3MPI3Win14Get_errhandlerEv, .Lfunc_end150-_ZNK3MPI3Win14Get_errhandlerEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE,"axG",@progbits,_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE,comdat
	.weak	_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE # -- Begin function _ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE,@function
_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE: # @_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$144, %rsp
	movq	32(%rbp), %rax
	movq	24(%rbp), %rax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -112(%rbp)                # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -100(%rbp)                # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	24(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	32(%rbp), %rdi
	callq	_ZNK3MPI2OpcvP9ompi_op_tEv
	movq	-112(%rbp), %rdi                # 8-byte Reload
	movl	-100(%rbp), %esi                # 4-byte Reload
	movq	-96(%rbp), %rdx                 # 8-byte Reload
	movl	-84(%rbp), %ecx                 # 4-byte Reload
	movq	-80(%rbp), %r8                  # 8-byte Reload
	movl	-68(%rbp), %r9d                 # 4-byte Reload
	movq	-64(%rbp), %r11                 # 8-byte Reload
	movq	%rax, %r10
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%r11, (%rsp)
	movq	%r10, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Accumulate@PLT
	addq	$144, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end151:
	.size	_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE, .Lfunc_end151-_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win8CompleteEv,"axG",@progbits,_ZNK3MPI3Win8CompleteEv,comdat
	.weak	_ZNK3MPI3Win8CompleteEv         # -- Begin function _ZNK3MPI3Win8CompleteEv
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win8CompleteEv,@function
_ZNK3MPI3Win8CompleteEv:                # @_ZNK3MPI3Win8CompleteEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	callq	MPI_Win_complete@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end152:
	.size	_ZNK3MPI3Win8CompleteEv, .Lfunc_end152-_ZNK3MPI3Win8CompleteEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win5FenceEi,"axG",@progbits,_ZNK3MPI3Win5FenceEi,comdat
	.weak	_ZNK3MPI3Win5FenceEi            # -- Begin function _ZNK3MPI3Win5FenceEi
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win5FenceEi,@function
_ZNK3MPI3Win5FenceEi:                   # @_ZNK3MPI3Win5FenceEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movq	8(%rax), %rsi
	callq	MPI_Win_fence@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end153:
	.size	_ZNK3MPI3Win5FenceEi, .Lfunc_end153-_ZNK3MPI3Win5FenceEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_,"axG",@progbits,_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_,comdat
	.weak	_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_ # -- Begin function _ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_,@function
_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_: # @_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	24(%rbp), %rax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -92(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %eax
	movl	%eax, -76(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	24(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movl	-92(%rbp), %esi                 # 4-byte Reload
	movq	-88(%rbp), %rdx                 # 8-byte Reload
	movl	-76(%rbp), %ecx                 # 4-byte Reload
	movq	-72(%rbp), %r8                  # 8-byte Reload
	movl	-60(%rbp), %r9d                 # 4-byte Reload
	movq	%rax, %r10
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	MPI_Get@PLT
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end154:
	.size	_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_, .Lfunc_end154-_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win9Get_groupEv,"axG",@progbits,_ZNK3MPI3Win9Get_groupEv,comdat
	.weak	_ZNK3MPI3Win9Get_groupEv        # -- Begin function _ZNK3MPI3Win9Get_groupEv
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win9Get_groupEv,@function
_ZNK3MPI3Win9Get_groupEv:               # @_ZNK3MPI3Win9Get_groupEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Win_get_group@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end155:
	.size	_ZNK3MPI3Win9Get_groupEv, .Lfunc_end155-_ZNK3MPI3Win9Get_groupEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win4LockEiii,"axG",@progbits,_ZNK3MPI3Win4LockEiii,comdat
	.weak	_ZNK3MPI3Win4LockEiii           # -- Begin function _ZNK3MPI3Win4LockEiii
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win4LockEiii,@function
_ZNK3MPI3Win4LockEiii:                  # @_ZNK3MPI3Win4LockEiii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movl	-16(%rbp), %esi
	movl	-20(%rbp), %edx
	movq	8(%rax), %rcx
	callq	MPI_Win_lock@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end156:
	.size	_ZNK3MPI3Win4LockEiii, .Lfunc_end156-_ZNK3MPI3Win4LockEiii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win4PostERKNS_5GroupEi,"axG",@progbits,_ZNK3MPI3Win4PostERKNS_5GroupEi,comdat
	.weak	_ZNK3MPI3Win4PostERKNS_5GroupEi # -- Begin function _ZNK3MPI3Win4PostERKNS_5GroupEi
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win4PostERKNS_5GroupEi,@function
_ZNK3MPI3Win4PostERKNS_5GroupEi:        # @_ZNK3MPI3Win4PostERKNS_5GroupEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI5GroupcvP12ompi_group_tEv
	movq	%rax, %rdi
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movl	-20(%rbp), %esi
	movq	8(%rax), %rdx
	callq	MPI_Win_post@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end157:
	.size	_ZNK3MPI3Win4PostERKNS_5GroupEi, .Lfunc_end157-_ZNK3MPI3Win4PostERKNS_5GroupEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_,"axG",@progbits,_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_,comdat
	.weak	_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_ # -- Begin function _ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_,@function
_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_: # @_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	24(%rbp), %rax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -92(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %eax
	movl	%eax, -76(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	movq	24(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movl	-92(%rbp), %esi                 # 4-byte Reload
	movq	-88(%rbp), %rdx                 # 8-byte Reload
	movl	-76(%rbp), %ecx                 # 4-byte Reload
	movq	-72(%rbp), %r8                  # 8-byte Reload
	movl	-60(%rbp), %r9d                 # 4-byte Reload
	movq	%rax, %r10
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rax
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	MPI_Put@PLT
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end158:
	.size	_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_, .Lfunc_end158-_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win5StartERKNS_5GroupEi,"axG",@progbits,_ZNK3MPI3Win5StartERKNS_5GroupEi,comdat
	.weak	_ZNK3MPI3Win5StartERKNS_5GroupEi # -- Begin function _ZNK3MPI3Win5StartERKNS_5GroupEi
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win5StartERKNS_5GroupEi,@function
_ZNK3MPI3Win5StartERKNS_5GroupEi:       # @_ZNK3MPI3Win5StartERKNS_5GroupEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdi
	callq	_ZNK3MPI5GroupcvP12ompi_group_tEv
	movq	%rax, %rdi
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movl	-20(%rbp), %esi
	movq	8(%rax), %rdx
	callq	MPI_Win_start@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end159:
	.size	_ZNK3MPI3Win5StartERKNS_5GroupEi, .Lfunc_end159-_ZNK3MPI3Win5StartERKNS_5GroupEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win4TestEv,"axG",@progbits,_ZNK3MPI3Win4TestEv,comdat
	.weak	_ZNK3MPI3Win4TestEv             # -- Begin function _ZNK3MPI3Win4TestEv
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win4TestEv,@function
_ZNK3MPI3Win4TestEv:                    # @_ZNK3MPI3Win4TestEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Win_test@PLT
	cmpl	$0, -12(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end160:
	.size	_ZNK3MPI3Win4TestEv, .Lfunc_end160-_ZNK3MPI3Win4TestEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win6UnlockEi,"axG",@progbits,_ZNK3MPI3Win6UnlockEi,comdat
	.weak	_ZNK3MPI3Win6UnlockEi           # -- Begin function _ZNK3MPI3Win6UnlockEi
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win6UnlockEi,@function
_ZNK3MPI3Win6UnlockEi:                  # @_ZNK3MPI3Win6UnlockEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edi
	movq	8(%rax), %rsi
	callq	MPI_Win_unlock@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end161:
	.size	_ZNK3MPI3Win6UnlockEi, .Lfunc_end161-_ZNK3MPI3Win6UnlockEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win4WaitEv,"axG",@progbits,_ZNK3MPI3Win4WaitEv,comdat
	.weak	_ZNK3MPI3Win4WaitEv             # -- Begin function _ZNK3MPI3Win4WaitEv
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win4WaitEv,@function
_ZNK3MPI3Win4WaitEv:                    # @_ZNK3MPI3Win4WaitEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	callq	MPI_Win_wait@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end162:
	.size	_ZNK3MPI3Win4WaitEv, .Lfunc_end162-_ZNK3MPI3Win4WaitEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win15Call_errhandlerEi,"axG",@progbits,_ZNK3MPI3Win15Call_errhandlerEi,comdat
	.weak	_ZNK3MPI3Win15Call_errhandlerEi # -- Begin function _ZNK3MPI3Win15Call_errhandlerEi
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win15Call_errhandlerEi,@function
_ZNK3MPI3Win15Call_errhandlerEi:        # @_ZNK3MPI3Win15Call_errhandlerEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	callq	MPI_Win_call_errhandler@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end163:
	.size	_ZNK3MPI3Win15Call_errhandlerEi, .Lfunc_end163-_ZNK3MPI3Win15Call_errhandlerEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI3Win11Delete_attrEi,"axG",@progbits,_ZN3MPI3Win11Delete_attrEi,comdat
	.weak	_ZN3MPI3Win11Delete_attrEi      # -- Begin function _ZN3MPI3Win11Delete_attrEi
	.p2align	4, 0x90
	.type	_ZN3MPI3Win11Delete_attrEi,@function
_ZN3MPI3Win11Delete_attrEi:             # @_ZN3MPI3Win11Delete_attrEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	callq	MPI_Win_delete_attr@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end164:
	.size	_ZN3MPI3Win11Delete_attrEi, .Lfunc_end164-_ZN3MPI3Win11Delete_attrEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI3Win8Get_nameEPcRi,"axG",@progbits,_ZNK3MPI3Win8Get_nameEPcRi,comdat
	.weak	_ZNK3MPI3Win8Get_nameEPcRi      # -- Begin function _ZNK3MPI3Win8Get_nameEPcRi
	.p2align	4, 0x90
	.type	_ZNK3MPI3Win8Get_nameEPcRi,@function
_ZNK3MPI3Win8Get_nameEPcRi:             # @_ZNK3MPI3Win8Get_nameEPcRi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Win_get_name@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end165:
	.size	_ZNK3MPI3Win8Get_nameEPcRi, .Lfunc_end165-_ZNK3MPI3Win8Get_nameEPcRi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI3Win8Set_attrEiPKv,"axG",@progbits,_ZN3MPI3Win8Set_attrEiPKv,comdat
	.weak	_ZN3MPI3Win8Set_attrEiPKv       # -- Begin function _ZN3MPI3Win8Set_attrEiPKv
	.p2align	4, 0x90
	.type	_ZN3MPI3Win8Set_attrEiPKv,@function
_ZN3MPI3Win8Set_attrEiPKv:              # @_ZN3MPI3Win8Set_attrEiPKv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	callq	MPI_Win_set_attr@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end166:
	.size	_ZN3MPI3Win8Set_attrEiPKv, .Lfunc_end166-_ZN3MPI3Win8Set_attrEiPKv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI3Win8Set_nameEPKc,"axG",@progbits,_ZN3MPI3Win8Set_nameEPKc,comdat
	.weak	_ZN3MPI3Win8Set_nameEPKc        # -- Begin function _ZN3MPI3Win8Set_nameEPKc
	.p2align	4, 0x90
	.type	_ZN3MPI3Win8Set_nameEPKc,@function
_ZN3MPI3Win8Set_nameEPKc:               # @_ZN3MPI3Win8Set_nameEPKc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	callq	MPI_Win_set_name@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end167:
	.size	_ZN3MPI3Win8Set_nameEPKc, .Lfunc_end167-_ZN3MPI3Win8Set_nameEPKc
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI10ErrhandlerD2Ev,"axG",@progbits,_ZN3MPI10ErrhandlerD2Ev,comdat
	.weak	_ZN3MPI10ErrhandlerD2Ev         # -- Begin function _ZN3MPI10ErrhandlerD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI10ErrhandlerD2Ev,@function
_ZN3MPI10ErrhandlerD2Ev:                # @_ZN3MPI10ErrhandlerD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end168:
	.size	_ZN3MPI10ErrhandlerD2Ev, .Lfunc_end168-_ZN3MPI10ErrhandlerD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI10ErrhandlerD0Ev,"axG",@progbits,_ZN3MPI10ErrhandlerD0Ev,comdat
	.weak	_ZN3MPI10ErrhandlerD0Ev         # -- Begin function _ZN3MPI10ErrhandlerD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI10ErrhandlerD0Ev,@function
_ZN3MPI10ErrhandlerD0Ev:                # @_ZN3MPI10ErrhandlerD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI10ErrhandlerD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end169:
	.size	_ZN3MPI10ErrhandlerD0Ev, .Lfunc_end169-_ZN3MPI10ErrhandlerD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9IntracommD2Ev,"axG",@progbits,_ZN3MPI9IntracommD2Ev,comdat
	.weak	_ZN3MPI9IntracommD2Ev           # -- Begin function _ZN3MPI9IntracommD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9IntracommD2Ev,@function
_ZN3MPI9IntracommD2Ev:                  # @_ZN3MPI9IntracommD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI4CommD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end170:
	.size	_ZN3MPI9IntracommD2Ev, .Lfunc_end170-_ZN3MPI9IntracommD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9IntracommD0Ev,"axG",@progbits,_ZN3MPI9IntracommD0Ev,comdat
	.weak	_ZN3MPI9IntracommD0Ev           # -- Begin function _ZN3MPI9IntracommD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9IntracommD0Ev,@function
_ZN3MPI9IntracommD0Ev:                  # @_ZN3MPI9IntracommD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI9IntracommD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end171:
	.size	_ZN3MPI9IntracommD0Ev, .Lfunc_end171-_ZN3MPI9IntracommD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm5CloneEv,"axG",@progbits,_ZNK3MPI9Intracomm5CloneEv,comdat
	.weak	_ZNK3MPI9Intracomm5CloneEv      # -- Begin function _ZNK3MPI9Intracomm5CloneEv
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm5CloneEv,@function
_ZNK3MPI9Intracomm5CloneEv:             # @_ZNK3MPI9Intracomm5CloneEv
.Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception5
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-16(%rbp), %rsi
	callq	MPI_Comm_dup@PLT
	movl	$16, %edi
	callq	_Znwm@PLT
	movq	%rax, %rdi
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rsi
.Ltmp19:
	callq	_ZN3MPI9IntracommC2EP19ompi_communicator_t
.Ltmp20:
	jmp	.LBB172_1
.LBB172_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB172_2:
	.cfi_def_cfa %rbp, 16
.Ltmp21:
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZdlPv@PLT
# %bb.3:
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end172:
	.size	_ZNK3MPI9Intracomm5CloneEv, .Lfunc_end172-_ZNK3MPI9Intracomm5CloneEv
	.cfi_endproc
	.section	.gcc_except_table._ZNK3MPI9Intracomm5CloneEv,"aG",@progbits,_ZNK3MPI9Intracomm5CloneEv,comdat
	.p2align	2
GCC_except_table172:
.Lexception5:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Lfunc_begin5-.Lfunc_begin5    # >> Call Site 1 <<
	.uleb128 .Ltmp19-.Lfunc_begin5          #   Call between .Lfunc_begin5 and .Ltmp19
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp19-.Lfunc_begin5          # >> Call Site 2 <<
	.uleb128 .Ltmp20-.Ltmp19                #   Call between .Ltmp19 and .Ltmp20
	.uleb128 .Ltmp21-.Lfunc_begin5          #     jumps to .Ltmp21
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp20-.Lfunc_begin5          # >> Call Site 3 <<
	.uleb128 .Lfunc_end172-.Ltmp20          #   Call between .Ltmp20 and .Lfunc_end172
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end5:
	.p2align	2
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE,"axG",@progbits,_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE,comdat
	.weak	_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE # -- Begin function _ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE,@function
_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE: # @_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI2OpcvP9ompi_op_tEv
	movq	-88(%rbp), %rdi                 # 8-byte Reload
	movq	-80(%rbp), %rsi                 # 8-byte Reload
	movl	-68(%rbp), %edx                 # 4-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %r8
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %r9
	callq	MPI_Exscan@PLT
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end173:
	.size	_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE, .Lfunc_end173-_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm6CreateERKNS_5GroupE,"axG",@progbits,_ZNK3MPI9Intracomm6CreateERKNS_5GroupE,comdat
	.weak	_ZNK3MPI9Intracomm6CreateERKNS_5GroupE # -- Begin function _ZNK3MPI9Intracomm6CreateERKNS_5GroupE
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm6CreateERKNS_5GroupE,@function
_ZNK3MPI9Intracomm6CreateERKNS_5GroupE: # @_ZNK3MPI9Intracomm6CreateERKNS_5GroupE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rdi
	callq	_ZNK3MPI5GroupcvP12ompi_group_tEv
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	leaq	-32(%rbp), %rdx
	callq	MPI_Comm_create@PLT
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	-32(%rbp), %rsi
	callq	_ZN3MPI9IntracommC2EP19ompi_communicator_t
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end174:
	.size	_ZNK3MPI9Intracomm6CreateERKNS_5GroupE, .Lfunc_end174-_ZNK3MPI9Intracomm6CreateERKNS_5GroupE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm5SplitEii,"axG",@progbits,_ZNK3MPI9Intracomm5SplitEii,comdat
	.weak	_ZNK3MPI9Intracomm5SplitEii     # -- Begin function _ZNK3MPI9Intracomm5SplitEii
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm5SplitEii,@function
_ZNK3MPI9Intracomm5SplitEii:            # @_ZNK3MPI9Intracomm5SplitEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movl	%ecx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movl	-24(%rbp), %edx
	leaq	-32(%rbp), %rcx
	callq	MPI_Comm_split@PLT
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	-32(%rbp), %rsi
	callq	_ZN3MPI9IntracommC2EP19ompi_communicator_t
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end175:
	.size	_ZNK3MPI9Intracomm5SplitEii, .Lfunc_end175-_ZNK3MPI9Intracomm5SplitEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii,"axG",@progbits,_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii,comdat
	.weak	_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii # -- Begin function _ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii,@function
_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii: # @_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -64(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	movq	-80(%rbp), %rdi                 # 8-byte Reload
	movl	-68(%rbp), %esi                 # 4-byte Reload
	movq	%rax, %rdx
	movl	-36(%rbp), %ecx
	movl	-40(%rbp), %r8d
	leaq	-48(%rbp), %r9
	callq	MPI_Intercomm_create@PLT
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movq	-48(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end176:
	.size	_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii, .Lfunc_end176-_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm11Create_cartEiPKiPKbb,"axG",@progbits,_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb,comdat
	.weak	_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb # -- Begin function _ZNK3MPI9Intracomm11Create_cartEiPKiPKbb
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb,@function
_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb: # @_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -96(%rbp)                 # 8-byte Spill
	movb	%r9b, %al
	movq	%rdi, %r9
	movq	%r9, -88(%rbp)                  # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	andb	$1, %al
	movb	%al, -41(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movslq	-20(%rbp), %rax
	movl	$4, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, -56(%rbp)
	movl	$0, -60(%rbp)
.LBB177_1:                              # =>This Inner Loop Header: Depth=1
	movl	-60(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.LBB177_4
# %bb.2:                                #   in Loop: Header=BB177_1 Depth=1
	movq	-40(%rbp), %rax
	movslq	-60(%rbp), %rcx
	movb	(%rax,%rcx), %al
	andb	$1, %al
	movzbl	%al, %edx
	movq	-56(%rbp), %rax
	movslq	-60(%rbp), %rcx
	movl	%edx, (%rax,%rcx,4)
# %bb.3:                                #   in Loop: Header=BB177_1 Depth=1
	movl	-60(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -60(%rbp)
	jmp	.LBB177_1
.LBB177_4:
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movq	-32(%rbp), %rdx
	movq	-56(%rbp), %rcx
	movb	-41(%rbp), %al
	andb	$1, %al
	movzbl	%al, %r8d
	leaq	-72(%rbp), %r9
	callq	MPI_Cart_create@PLT
	movq	-56(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB177_6
# %bb.5:
	movq	-104(%rbp), %rdi                # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB177_6:
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	leaq	-72(%rbp), %rsi
	callq	_ZN3MPI8CartcommC2ERKP19ompi_communicator_t
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end177:
	.size	_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb, .Lfunc_end177-_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm12Create_graphEiPKiS2_b,"axG",@progbits,_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b,comdat
	.weak	_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b # -- Begin function _ZNK3MPI9Intracomm12Create_graphEiPKiS2_b
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b,@function
_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b: # @_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movb	%r9b, %al
	movq	%rdi, %r9
	movq	%r9, -64(%rbp)                  # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	andb	$1, %al
	movb	%al, -41(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movb	-41(%rbp), %al
	andb	$1, %al
	movzbl	%al, %r8d
	leaq	-56(%rbp), %r9
	callq	MPI_Graph_create@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI9GraphcommC2ERKP19ompi_communicator_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end178:
	.size	_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b, .Lfunc_end178-_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi,"axG",@progbits,_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi,comdat
	.weak	_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi # -- Begin function _ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi,@function
_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi: # @_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -64(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI4InfocvP11ompi_info_tEv
	movq	-80(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %edx
	movq	8(%rax), %rcx
	leaq	-48(%rbp), %r8
	callq	MPI_Comm_accept@PLT
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movq	-48(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end179:
	.size	_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi, .Lfunc_end179-_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi,"axG",@progbits,_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi,comdat
	.weak	_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi # -- Begin function _ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi,@function
_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi: # @_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -64(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-32(%rbp), %rdi
	callq	_ZNK3MPI4InfocvP11ompi_info_tEv
	movq	-80(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movl	-36(%rbp), %edx
	movq	8(%rax), %rcx
	leaq	-48(%rbp), %r8
	callq	MPI_Comm_connect@PLT
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	movq	-48(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end180:
	.size	_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi, .Lfunc_end180-_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi,"axG",@progbits,_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi,comdat
	.weak	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi # -- Begin function _ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi,@function
_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi: # @_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movq	-32(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI4InfocvP11ompi_info_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movq	-96(%rbp), %rsi                 # 8-byte Reload
	movl	-84(%rbp), %edx                 # 4-byte Reload
	movq	%rax, %rcx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	leaq	-56(%rbp), %rax
	xorl	%r10d, %r10d
                                        # kill: def $r10 killed $r10d
	movq	%rax, (%rsp)
	movq	$0, 8(%rsp)
	callq	MPI_Comm_spawn@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end181:
	.size	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi, .Lfunc_end181-_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi,"axG",@progbits,_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi,comdat
	.weak	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi # -- Begin function _ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi,@function
_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi: # @_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	%rdi, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	24(%rbp), %rax
	movl	16(%rbp), %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movq	%r9, -48(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -104(%rbp)                # 8-byte Spill
	movq	-32(%rbp), %rax
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movl	-36(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	movq	-48(%rbp), %rdi
	callq	_ZNK3MPI4InfocvP11ompi_info_tEv
	movq	-104(%rbp), %rdi                # 8-byte Reload
	movq	-96(%rbp), %rsi                 # 8-byte Reload
	movl	-84(%rbp), %edx                 # 4-byte Reload
	movq	%rax, %rcx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	16(%rbp), %r8d
	movq	8(%rax), %r9
	movq	24(%rbp), %rax
	leaq	-56(%rbp), %r10
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	callq	MPI_Comm_spawn@PLT
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end182:
	.size	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi, .Lfunc_end182-_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi,"axG",@progbits,_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi,comdat
	.weak	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi # -- Begin function _ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi
	.p2align	4, 0x90
	.type	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi,@function
_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi: # @_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	%rdi, -96(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movl	24(%rbp), %eax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %edi
	movq	16(%rbp), %rsi
	callq	_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE
	movq	%rax, %rcx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movq	%rcx, -64(%rbp)
	movl	-20(%rbp), %edi
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rdx
	movq	-48(%rbp), %rcx
	movq	-64(%rbp), %r8
	movl	24(%rbp), %r9d
	movq	8(%rax), %r10
	leaq	-56(%rbp), %rax
	xorl	%r11d, %r11d
                                        # kill: def $r11 killed $r11d
	movq	%r10, (%rsp)
	movq	%rax, 8(%rsp)
	movq	$0, 16(%rsp)
	callq	MPI_Comm_spawn_multiple@PLT
	movq	-64(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB183_2
# %bb.1:
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB183_2:
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end183:
	.size	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi, .Lfunc_end183-_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi,"axG",@progbits,_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi,comdat
	.weak	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi # -- Begin function _ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi
	.p2align	4, 0x90
	.type	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi,@function
_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi: # @_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$128, %rsp
	movq	%rdi, -96(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	movq	32(%rbp), %rax
	movl	24(%rbp), %eax
	movq	16(%rbp), %rax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	%r9, -48(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	-20(%rbp), %edi
	movq	16(%rbp), %rsi
	callq	_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE
	movq	%rax, %rcx
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movq	%rcx, -64(%rbp)
	movl	-20(%rbp), %edi
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rdx
	movq	-48(%rbp), %rcx
	movq	-64(%rbp), %r8
	movl	24(%rbp), %r9d
	movq	8(%rax), %r11
	movq	32(%rbp), %rax
	leaq	-56(%rbp), %r10
	movq	%r11, (%rsp)
	movq	%r10, 8(%rsp)
	movq	%rax, 16(%rsp)
	callq	MPI_Comm_spawn_multiple@PLT
	movq	-64(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB184_2
# %bb.1:
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB184_2:
	movq	-96(%rbp), %rdi                 # 8-byte Reload
	movq	-56(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end184:
	.size	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi, .Lfunc_end184-_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8CartcommD2Ev,"axG",@progbits,_ZN3MPI8CartcommD2Ev,comdat
	.weak	_ZN3MPI8CartcommD2Ev            # -- Begin function _ZN3MPI8CartcommD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8CartcommD2Ev,@function
_ZN3MPI8CartcommD2Ev:                   # @_ZN3MPI8CartcommD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI9IntracommD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end185:
	.size	_ZN3MPI8CartcommD2Ev, .Lfunc_end185-_ZN3MPI8CartcommD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8CartcommD0Ev,"axG",@progbits,_ZN3MPI8CartcommD0Ev,comdat
	.weak	_ZN3MPI8CartcommD0Ev            # -- Begin function _ZN3MPI8CartcommD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI8CartcommD0Ev,@function
_ZN3MPI8CartcommD0Ev:                   # @_ZN3MPI8CartcommD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI8CartcommD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end186:
	.size	_ZN3MPI8CartcommD0Ev, .Lfunc_end186-_ZN3MPI8CartcommD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm7Get_dimEv,"axG",@progbits,_ZNK3MPI8Cartcomm7Get_dimEv,comdat
	.weak	_ZNK3MPI8Cartcomm7Get_dimEv     # -- Begin function _ZNK3MPI8Cartcomm7Get_dimEv
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm7Get_dimEv,@function
_ZNK3MPI8Cartcomm7Get_dimEv:            # @_ZNK3MPI8Cartcomm7Get_dimEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Cartdim_get@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end187:
	.size	_ZNK3MPI8Cartcomm7Get_dimEv, .Lfunc_end187-_ZNK3MPI8Cartcomm7Get_dimEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_,"axG",@progbits,_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_,comdat
	.weak	_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_ # -- Begin function _ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_,@function
_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_:    # @_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movslq	-12(%rbp), %rax
	movl	$4, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, -48(%rbp)
	movl	$0, -52(%rbp)
.LBB188_1:                              # =>This Inner Loop Header: Depth=1
	movl	-52(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jge	.LBB188_4
# %bb.2:                                #   in Loop: Header=BB188_1 Depth=1
	movq	-32(%rbp), %rax
	movslq	-52(%rbp), %rcx
	movb	(%rax,%rcx), %al
	andb	$1, %al
	movzbl	%al, %edx
	movq	-48(%rbp), %rax
	movslq	-52(%rbp), %rcx
	movl	%edx, (%rax,%rcx,4)
# %bb.3:                                #   in Loop: Header=BB188_1 Depth=1
	movl	-52(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -52(%rbp)
	jmp	.LBB188_1
.LBB188_4:
	movq	-64(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	movq	-48(%rbp), %rcx
	movq	-40(%rbp), %r8
	callq	MPI_Cart_get@PLT
	movl	$0, -52(%rbp)
.LBB188_5:                              # =>This Inner Loop Header: Depth=1
	movl	-52(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jge	.LBB188_8
# %bb.6:                                #   in Loop: Header=BB188_5 Depth=1
	movq	-48(%rbp), %rax
	movslq	-52(%rbp), %rcx
	cmpl	$0, (%rax,%rcx,4)
	setne	%dl
	movq	-32(%rbp), %rax
	movslq	-52(%rbp), %rcx
	andb	$1, %dl
	movb	%dl, (%rax,%rcx)
# %bb.7:                                #   in Loop: Header=BB188_5 Depth=1
	movl	-52(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -52(%rbp)
	jmp	.LBB188_5
.LBB188_8:
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB188_10
# %bb.9:
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB188_10:
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end188:
	.size	_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_, .Lfunc_end188-_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm13Get_cart_rankEPKi,"axG",@progbits,_ZNK3MPI8Cartcomm13Get_cart_rankEPKi,comdat
	.weak	_ZNK3MPI8Cartcomm13Get_cart_rankEPKi # -- Begin function _ZNK3MPI8Cartcomm13Get_cart_rankEPKi
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm13Get_cart_rankEPKi,@function
_ZNK3MPI8Cartcomm13Get_cart_rankEPKi:   # @_ZNK3MPI8Cartcomm13Get_cart_rankEPKi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	leaq	-20(%rbp), %rdx
	callq	MPI_Cart_rank@PLT
	movl	-20(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end189:
	.size	_ZNK3MPI8Cartcomm13Get_cart_rankEPKi, .Lfunc_end189-_ZNK3MPI8Cartcomm13Get_cart_rankEPKi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm10Get_coordsEiiPi,"axG",@progbits,_ZNK3MPI8Cartcomm10Get_coordsEiiPi,comdat
	.weak	_ZNK3MPI8Cartcomm10Get_coordsEiiPi # -- Begin function _ZNK3MPI8Cartcomm10Get_coordsEiiPi
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm10Get_coordsEiiPi,@function
_ZNK3MPI8Cartcomm10Get_coordsEiiPi:     # @_ZNK3MPI8Cartcomm10Get_coordsEiiPi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movl	-16(%rbp), %edx
	movq	-24(%rbp), %rcx
	callq	MPI_Cart_coords@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end190:
	.size	_ZNK3MPI8Cartcomm10Get_coordsEiiPi, .Lfunc_end190-_ZNK3MPI8Cartcomm10Get_coordsEiiPi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm5ShiftEiiRiS1_,"axG",@progbits,_ZNK3MPI8Cartcomm5ShiftEiiRiS1_,comdat
	.weak	_ZNK3MPI8Cartcomm5ShiftEiiRiS1_ # -- Begin function _ZNK3MPI8Cartcomm5ShiftEiiRiS1_
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm5ShiftEiiRiS1_,@function
_ZNK3MPI8Cartcomm5ShiftEiiRiS1_:        # @_ZNK3MPI8Cartcomm5ShiftEiiRiS1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	%r8, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movl	-16(%rbp), %edx
	movq	-24(%rbp), %rcx
	movq	-32(%rbp), %r8
	callq	MPI_Cart_shift@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end191:
	.size	_ZNK3MPI8Cartcomm5ShiftEiiRiS1_, .Lfunc_end191-_ZNK3MPI8Cartcomm5ShiftEiiRiS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm3SubEPKb,"axG",@progbits,_ZNK3MPI8Cartcomm3SubEPKb,comdat
	.weak	_ZNK3MPI8Cartcomm3SubEPKb       # -- Begin function _ZNK3MPI8Cartcomm3SubEPKb
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm3SubEPKb,@function
_ZNK3MPI8Cartcomm3SubEPKb:              # @_ZNK3MPI8Cartcomm3SubEPKb
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$96, %rsp
	movq	%rdi, -80(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	8(%rax), %rdi
	leaq	-28(%rbp), %rsi
	callq	MPI_Cartdim_get@PLT
	movslq	-28(%rbp), %rax
	movl	$4, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, -40(%rbp)
	movl	$0, -44(%rbp)
.LBB192_1:                              # =>This Inner Loop Header: Depth=1
	movl	-44(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.LBB192_4
# %bb.2:                                #   in Loop: Header=BB192_1 Depth=1
	movq	-24(%rbp), %rax
	movslq	-44(%rbp), %rcx
	movb	(%rax,%rcx), %al
	andb	$1, %al
	movzbl	%al, %edx
	movq	-40(%rbp), %rax
	movslq	-44(%rbp), %rcx
	movl	%edx, (%rax,%rcx,4)
# %bb.3:                                #   in Loop: Header=BB192_1 Depth=1
	movl	-44(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -44(%rbp)
	jmp	.LBB192_1
.LBB192_4:
	movq	-64(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rdi
	movq	-40(%rbp), %rsi
	leaq	-56(%rbp), %rdx
	callq	MPI_Cart_sub@PLT
	movq	-40(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB192_6
# %bb.5:
	movq	-88(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB192_6:
	movq	-80(%rbp), %rdi                 # 8-byte Reload
	leaq	-56(%rbp), %rsi
	callq	_ZN3MPI8CartcommC2ERKP19ompi_communicator_t
	movq	-72(%rbp), %rax                 # 8-byte Reload
	addq	$96, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end192:
	.size	_ZNK3MPI8Cartcomm3SubEPKb, .Lfunc_end192-_ZNK3MPI8Cartcomm3SubEPKb
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI8Cartcomm3MapEiPKiPKb,"axG",@progbits,_ZNK3MPI8Cartcomm3MapEiPKiPKb,comdat
	.weak	_ZNK3MPI8Cartcomm3MapEiPKiPKb   # -- Begin function _ZNK3MPI8Cartcomm3MapEiPKiPKb
	.p2align	4, 0x90
	.type	_ZNK3MPI8Cartcomm3MapEiPKiPKb,@function
_ZNK3MPI8Cartcomm3MapEiPKiPKb:          # @_ZNK3MPI8Cartcomm3MapEiPKiPKb
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movslq	-12(%rbp), %rax
	movl	$4, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, -40(%rbp)
	movl	$0, -44(%rbp)
.LBB193_1:                              # =>This Inner Loop Header: Depth=1
	movl	-44(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jge	.LBB193_4
# %bb.2:                                #   in Loop: Header=BB193_1 Depth=1
	movq	-32(%rbp), %rax
	movslq	-44(%rbp), %rcx
	movb	(%rax,%rcx), %al
	andb	$1, %al
	movzbl	%al, %edx
	movq	-40(%rbp), %rax
	movslq	-44(%rbp), %rcx
	movl	%edx, (%rax,%rcx,4)
# %bb.3:                                #   in Loop: Header=BB193_1 Depth=1
	movl	-44(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -44(%rbp)
	jmp	.LBB193_1
.LBB193_4:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	movq	-40(%rbp), %rcx
	leaq	-48(%rbp), %r8
	callq	MPI_Cart_map@PLT
	movq	-40(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB193_6
# %bb.5:
	movq	-64(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdaPv@PLT
.LBB193_6:
	movl	-48(%rbp), %eax
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end193:
	.size	_ZNK3MPI8Cartcomm3MapEiPKiPKb, .Lfunc_end193-_ZNK3MPI8Cartcomm3MapEiPKiPKb
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9GraphcommD2Ev,"axG",@progbits,_ZN3MPI9GraphcommD2Ev,comdat
	.weak	_ZN3MPI9GraphcommD2Ev           # -- Begin function _ZN3MPI9GraphcommD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9GraphcommD2Ev,@function
_ZN3MPI9GraphcommD2Ev:                  # @_ZN3MPI9GraphcommD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI9IntracommD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end194:
	.size	_ZN3MPI9GraphcommD2Ev, .Lfunc_end194-_ZN3MPI9GraphcommD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9GraphcommD0Ev,"axG",@progbits,_ZN3MPI9GraphcommD0Ev,comdat
	.weak	_ZN3MPI9GraphcommD0Ev           # -- Begin function _ZN3MPI9GraphcommD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9GraphcommD0Ev,@function
_ZN3MPI9GraphcommD0Ev:                  # @_ZN3MPI9GraphcommD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI9GraphcommD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end195:
	.size	_ZN3MPI9GraphcommD0Ev, .Lfunc_end195-_ZN3MPI9GraphcommD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Graphcomm8Get_dimsEPiS1_,"axG",@progbits,_ZNK3MPI9Graphcomm8Get_dimsEPiS1_,comdat
	.weak	_ZNK3MPI9Graphcomm8Get_dimsEPiS1_ # -- Begin function _ZNK3MPI9Graphcomm8Get_dimsEPiS1_
	.p2align	4, 0x90
	.type	_ZNK3MPI9Graphcomm8Get_dimsEPiS1_,@function
_ZNK3MPI9Graphcomm8Get_dimsEPiS1_:      # @_ZNK3MPI9Graphcomm8Get_dimsEPiS1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Graphdims_get@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end196:
	.size	_ZNK3MPI9Graphcomm8Get_dimsEPiS1_, .Lfunc_end196-_ZNK3MPI9Graphcomm8Get_dimsEPiS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Graphcomm8Get_topoEiiPiS1_,"axG",@progbits,_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_,comdat
	.weak	_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_ # -- Begin function _ZNK3MPI9Graphcomm8Get_topoEiiPiS1_
	.p2align	4, 0x90
	.type	_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_,@function
_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_:    # @_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	%r8, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movl	-16(%rbp), %edx
	movq	-24(%rbp), %rcx
	movq	-32(%rbp), %r8
	callq	MPI_Graph_get@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end197:
	.size	_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_, .Lfunc_end197-_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Graphcomm19Get_neighbors_countEi,"axG",@progbits,_ZNK3MPI9Graphcomm19Get_neighbors_countEi,comdat
	.weak	_ZNK3MPI9Graphcomm19Get_neighbors_countEi # -- Begin function _ZNK3MPI9Graphcomm19Get_neighbors_countEi
	.p2align	4, 0x90
	.type	_ZNK3MPI9Graphcomm19Get_neighbors_countEi,@function
_ZNK3MPI9Graphcomm19Get_neighbors_countEi: # @_ZNK3MPI9Graphcomm19Get_neighbors_countEi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	leaq	-16(%rbp), %rdx
	callq	MPI_Graph_neighbors_count@PLT
	movl	-16(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end198:
	.size	_ZNK3MPI9Graphcomm19Get_neighbors_countEi, .Lfunc_end198-_ZNK3MPI9Graphcomm19Get_neighbors_countEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Graphcomm13Get_neighborsEiiPi,"axG",@progbits,_ZNK3MPI9Graphcomm13Get_neighborsEiiPi,comdat
	.weak	_ZNK3MPI9Graphcomm13Get_neighborsEiiPi # -- Begin function _ZNK3MPI9Graphcomm13Get_neighborsEiiPi
	.p2align	4, 0x90
	.type	_ZNK3MPI9Graphcomm13Get_neighborsEiiPi,@function
_ZNK3MPI9Graphcomm13Get_neighborsEiiPi: # @_ZNK3MPI9Graphcomm13Get_neighborsEiiPi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movl	-16(%rbp), %edx
	movq	-24(%rbp), %rcx
	callq	MPI_Graph_neighbors@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end199:
	.size	_ZNK3MPI9Graphcomm13Get_neighborsEiiPi, .Lfunc_end199-_ZNK3MPI9Graphcomm13Get_neighborsEiiPi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Graphcomm3MapEiPKiS2_,"axG",@progbits,_ZNK3MPI9Graphcomm3MapEiPKiS2_,comdat
	.weak	_ZNK3MPI9Graphcomm3MapEiPKiS2_  # -- Begin function _ZNK3MPI9Graphcomm3MapEiPKiS2_
	.p2align	4, 0x90
	.type	_ZNK3MPI9Graphcomm3MapEiPKiS2_,@function
_ZNK3MPI9Graphcomm3MapEiPKiS2_:         # @_ZNK3MPI9Graphcomm3MapEiPKiS2_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rcx
	leaq	-36(%rbp), %r8
	callq	MPI_Graph_map@PLT
	movl	-36(%rbp), %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end200:
	.size	_ZNK3MPI9Graphcomm3MapEiPKiS2_, .Lfunc_end200-_ZNK3MPI9Graphcomm3MapEiPKiS2_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9IntercommD2Ev,"axG",@progbits,_ZN3MPI9IntercommD2Ev,comdat
	.weak	_ZN3MPI9IntercommD2Ev           # -- Begin function _ZN3MPI9IntercommD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9IntercommD2Ev,@function
_ZN3MPI9IntercommD2Ev:                  # @_ZN3MPI9IntercommD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN3MPI4CommD2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end201:
	.size	_ZN3MPI9IntercommD2Ev, .Lfunc_end201-_ZN3MPI9IntercommD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9IntercommD0Ev,"axG",@progbits,_ZN3MPI9IntercommD0Ev,comdat
	.weak	_ZN3MPI9IntercommD0Ev           # -- Begin function _ZN3MPI9IntercommD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9IntercommD0Ev,@function
_ZN3MPI9IntercommD0Ev:                  # @_ZN3MPI9IntercommD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI9IntercommD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end202:
	.size	_ZN3MPI9IntercommD0Ev, .Lfunc_end202-_ZN3MPI9IntercommD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intercomm15Get_remote_sizeEv,"axG",@progbits,_ZNK3MPI9Intercomm15Get_remote_sizeEv,comdat
	.weak	_ZNK3MPI9Intercomm15Get_remote_sizeEv # -- Begin function _ZNK3MPI9Intercomm15Get_remote_sizeEv
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intercomm15Get_remote_sizeEv,@function
_ZNK3MPI9Intercomm15Get_remote_sizeEv:  # @_ZNK3MPI9Intercomm15Get_remote_sizeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Comm_remote_size@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end203:
	.size	_ZNK3MPI9Intercomm15Get_remote_sizeEv, .Lfunc_end203-_ZNK3MPI9Intercomm15Get_remote_sizeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intercomm16Get_remote_groupEv,"axG",@progbits,_ZNK3MPI9Intercomm16Get_remote_groupEv,comdat
	.weak	_ZNK3MPI9Intercomm16Get_remote_groupEv # -- Begin function _ZNK3MPI9Intercomm16Get_remote_groupEv
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intercomm16Get_remote_groupEv,@function
_ZNK3MPI9Intercomm16Get_remote_groupEv: # @_ZNK3MPI9Intercomm16Get_remote_groupEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Comm_remote_group@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI5GroupC2EP12ompi_group_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end204:
	.size	_ZNK3MPI9Intercomm16Get_remote_groupEv, .Lfunc_end204-_ZNK3MPI9Intercomm16Get_remote_groupEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intercomm5MergeEb,"axG",@progbits,_ZNK3MPI9Intercomm5MergeEb,comdat
	.weak	_ZNK3MPI9Intercomm5MergeEb      # -- Begin function _ZNK3MPI9Intercomm5MergeEb
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intercomm5MergeEb,@function
_ZNK3MPI9Intercomm5MergeEb:             # @_ZNK3MPI9Intercomm5MergeEb
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	movb	%dl, %al
	movq	%rdi, %rcx
	movq	%rcx, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	andb	$1, %al
	movb	%al, -17(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movb	-17(%rbp), %al
	andb	$1, %al
	movzbl	%al, %esi
	leaq	-32(%rbp), %rdx
	callq	MPI_Intercomm_merge@PLT
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	-32(%rbp), %rsi
	callq	_ZN3MPI9IntracommC2EP19ompi_communicator_t
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end205:
	.size	_ZNK3MPI9Intercomm5MergeEb, .Lfunc_end205-_ZNK3MPI9Intercomm5MergeEb
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intercomm6CreateERKNS_5GroupE,"axG",@progbits,_ZNK3MPI9Intercomm6CreateERKNS_5GroupE,comdat
	.weak	_ZNK3MPI9Intercomm6CreateERKNS_5GroupE # -- Begin function _ZNK3MPI9Intercomm6CreateERKNS_5GroupE
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intercomm6CreateERKNS_5GroupE,@function
_ZNK3MPI9Intercomm6CreateERKNS_5GroupE: # @_ZNK3MPI9Intercomm6CreateERKNS_5GroupE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rdi
	callq	_ZNK3MPI5GroupcvP12ompi_group_tEv
	movq	-56(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rsi
	leaq	-32(%rbp), %rdx
	callq	MPI_Comm_create@PLT
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	-32(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end206:
	.size	_ZNK3MPI9Intercomm6CreateERKNS_5GroupE, .Lfunc_end206-_ZNK3MPI9Intercomm6CreateERKNS_5GroupE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Intercomm5SplitEii,"axG",@progbits,_ZNK3MPI9Intercomm5SplitEii,comdat
	.weak	_ZNK3MPI9Intercomm5SplitEii     # -- Begin function _ZNK3MPI9Intercomm5SplitEii
	.p2align	4, 0x90
	.type	_ZNK3MPI9Intercomm5SplitEii,@function
_ZNK3MPI9Intercomm5SplitEii:            # @_ZNK3MPI9Intercomm5SplitEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movl	%ecx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-20(%rbp), %esi
	movl	-24(%rbp), %edx
	leaq	-32(%rbp), %rcx
	callq	MPI_Comm_split@PLT
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	-32(%rbp), %rsi
	callq	_ZN3MPI9IntercommC2EP19ompi_communicator_t
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end207:
	.size	_ZNK3MPI9Intercomm5SplitEii, .Lfunc_end207-_ZNK3MPI9Intercomm5SplitEii
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4InfoD2Ev,"axG",@progbits,_ZN3MPI4InfoD2Ev,comdat
	.weak	_ZN3MPI4InfoD2Ev                # -- Begin function _ZN3MPI4InfoD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI4InfoD2Ev,@function
_ZN3MPI4InfoD2Ev:                       # @_ZN3MPI4InfoD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end208:
	.size	_ZN3MPI4InfoD2Ev, .Lfunc_end208-_ZN3MPI4InfoD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4InfoD0Ev,"axG",@progbits,_ZN3MPI4InfoD0Ev,comdat
	.weak	_ZN3MPI4InfoD0Ev                # -- Begin function _ZN3MPI4InfoD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI4InfoD0Ev,@function
_ZN3MPI4InfoD0Ev:                       # @_ZN3MPI4InfoD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI4InfoD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end209:
	.size	_ZN3MPI4InfoD0Ev, .Lfunc_end209-_ZN3MPI4InfoD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Info3DupEv,"axG",@progbits,_ZNK3MPI4Info3DupEv,comdat
	.weak	_ZNK3MPI4Info3DupEv             # -- Begin function _ZNK3MPI4Info3DupEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Info3DupEv,@function
_ZNK3MPI4Info3DupEv:                    # @_ZNK3MPI4Info3DupEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)                 # 8-byte Spill
	movq	%rdi, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-24(%rbp), %rsi
	callq	MPI_Info_dup@PLT
	movq	-40(%rbp), %rdi                 # 8-byte Reload
	movq	-24(%rbp), %rsi
	callq	_ZN3MPI4InfoC2EP11ompi_info_t
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end210:
	.size	_ZNK3MPI4Info3DupEv, .Lfunc_end210-_ZNK3MPI4Info3DupEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Info4FreeEv,"axG",@progbits,_ZN3MPI4Info4FreeEv,comdat
	.weak	_ZN3MPI4Info4FreeEv             # -- Begin function _ZN3MPI4Info4FreeEv
	.p2align	4, 0x90
	.type	_ZN3MPI4Info4FreeEv,@function
_ZN3MPI4Info4FreeEv:                    # @_ZN3MPI4Info4FreeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Info_free@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end211:
	.size	_ZN3MPI4Info4FreeEv, .Lfunc_end211-_ZN3MPI4Info4FreeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Info3GetEPKciPc,"axG",@progbits,_ZNK3MPI4Info3GetEPKciPc,comdat
	.weak	_ZNK3MPI4Info3GetEPKciPc        # -- Begin function _ZNK3MPI4Info3GetEPKciPc
	.p2align	4, 0x90
	.type	_ZNK3MPI4Info3GetEPKciPc,@function
_ZNK3MPI4Info3GetEPKciPc:               # @_ZNK3MPI4Info3GetEPKciPc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movl	-20(%rbp), %edx
	movq	-32(%rbp), %rcx
	leaq	-36(%rbp), %r8
	callq	MPI_Info_get@PLT
	cmpl	$0, -36(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end212:
	.size	_ZNK3MPI4Info3GetEPKciPc, .Lfunc_end212-_ZNK3MPI4Info3GetEPKciPc
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Info9Get_nkeysEv,"axG",@progbits,_ZNK3MPI4Info9Get_nkeysEv,comdat
	.weak	_ZNK3MPI4Info9Get_nkeysEv       # -- Begin function _ZNK3MPI4Info9Get_nkeysEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4Info9Get_nkeysEv,@function
_ZNK3MPI4Info9Get_nkeysEv:              # @_ZNK3MPI4Info9Get_nkeysEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Info_get_nkeys@PLT
	movl	-12(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end213:
	.size	_ZNK3MPI4Info9Get_nkeysEv, .Lfunc_end213-_ZNK3MPI4Info9Get_nkeysEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Info10Get_nthkeyEiPc,"axG",@progbits,_ZNK3MPI4Info10Get_nthkeyEiPc,comdat
	.weak	_ZNK3MPI4Info10Get_nthkeyEiPc   # -- Begin function _ZNK3MPI4Info10Get_nthkeyEiPc
	.p2align	4, 0x90
	.type	_ZNK3MPI4Info10Get_nthkeyEiPc,@function
_ZNK3MPI4Info10Get_nthkeyEiPc:          # @_ZNK3MPI4Info10Get_nthkeyEiPc
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movl	-12(%rbp), %esi
	movq	-24(%rbp), %rdx
	callq	MPI_Info_get_nthkey@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end214:
	.size	_ZNK3MPI4Info10Get_nthkeyEiPc, .Lfunc_end214-_ZNK3MPI4Info10Get_nthkeyEiPc
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI4Info12Get_valuelenEPKcRi,"axG",@progbits,_ZNK3MPI4Info12Get_valuelenEPKcRi,comdat
	.weak	_ZNK3MPI4Info12Get_valuelenEPKcRi # -- Begin function _ZNK3MPI4Info12Get_valuelenEPKcRi
	.p2align	4, 0x90
	.type	_ZNK3MPI4Info12Get_valuelenEPKcRi,@function
_ZNK3MPI4Info12Get_valuelenEPKcRi:      # @_ZNK3MPI4Info12Get_valuelenEPKcRi
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rcx
	callq	MPI_Info_get_valuelen@PLT
	cmpl	$0, -28(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end215:
	.size	_ZNK3MPI4Info12Get_valuelenEPKcRi, .Lfunc_end215-_ZNK3MPI4Info12Get_valuelenEPKcRi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4Info3SetEPKcS2_,"axG",@progbits,_ZN3MPI4Info3SetEPKcS2_,comdat
	.weak	_ZN3MPI4Info3SetEPKcS2_         # -- Begin function _ZN3MPI4Info3SetEPKcS2_
	.p2align	4, 0x90
	.type	_ZN3MPI4Info3SetEPKcS2_,@function
_ZN3MPI4Info3SetEPKcS2_:                # @_ZN3MPI4Info3SetEPKcS2_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	movq	-24(%rbp), %rdx
	callq	MPI_Info_set@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end216:
	.size	_ZN3MPI4Info3SetEPKcS2_, .Lfunc_end216-_ZN3MPI4Info3SetEPKcS2_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb,"axG",@progbits,_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb,comdat
	.weak	_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb # -- Begin function _ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb
	.p2align	4, 0x90
	.type	_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb,@function
_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb: # @_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movb	%dl, %al
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	andb	$1, %al
	movb	%al, -17(%rbp)
	movq	-8(%rbp), %rdx
	movq	%rdx, -32(%rbp)                 # 8-byte Spill
	movb	-17(%rbp), %al
	andb	$1, %al
	movzbl	%al, %esi
	addq	$8, %rdx
	movq	ompi_mpi_cxx_op_intercept@GOTPCREL(%rip), %rdi
	callq	MPI_Op_create@PLT
                                        # kill: def $ecx killed $eax
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %rdi
	movq	-16(%rbp), %rsi
	callq	ompi_op_set_cxx_callback@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end217:
	.size	_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb, .Lfunc_end217-_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI2Op4FreeEv,"axG",@progbits,_ZN3MPI2Op4FreeEv,comdat
	.weak	_ZN3MPI2Op4FreeEv               # -- Begin function _ZN3MPI2Op4FreeEv
	.p2align	4, 0x90
	.type	_ZN3MPI2Op4FreeEv,@function
_ZN3MPI2Op4FreeEv:                      # @_ZN3MPI2Op4FreeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	addq	$8, %rdi
	callq	MPI_Op_free@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end218:
	.size	_ZN3MPI2Op4FreeEv, .Lfunc_end218-_ZN3MPI2Op4FreeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE,"axG",@progbits,_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE,comdat
	.weak	_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE # -- Begin function _ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE
	.p2align	4, 0x90
	.type	_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE,@function
_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE: # @_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movq	%r8, -40(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	-28(%rbp), %eax
	movl	%eax, -52(%rbp)                 # 4-byte Spill
	movq	-40(%rbp), %rdi
	callq	_ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	movq	-72(%rbp), %rdi                 # 8-byte Reload
	movq	-64(%rbp), %rsi                 # 8-byte Reload
	movl	-52(%rbp), %edx                 # 4-byte Reload
	movq	%rax, %rcx
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	8(%rax), %r8
	callq	MPI_Reduce_local@PLT
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end219:
	.size	_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE, .Lfunc_end219-_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI2Op14Is_commutativeEv,"axG",@progbits,_ZNK3MPI2Op14Is_commutativeEv,comdat
	.weak	_ZNK3MPI2Op14Is_commutativeEv   # -- Begin function _ZNK3MPI2Op14Is_commutativeEv
	.p2align	4, 0x90
	.type	_ZNK3MPI2Op14Is_commutativeEv,@function
_ZNK3MPI2Op14Is_commutativeEv:          # @_ZNK3MPI2Op14Is_commutativeEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdi
	leaq	-12(%rbp), %rsi
	callq	MPI_Op_commutative@PLT
	cmpl	$0, -12(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end220:
	.size	_ZNK3MPI2Op14Is_commutativeEv, .Lfunc_end220-_ZNK3MPI2Op14Is_commutativeEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9IntracommC2Ev,"axG",@progbits,_ZN3MPI9IntracommC2Ev,comdat
	.weak	_ZN3MPI9IntracommC2Ev           # -- Begin function _ZN3MPI9IntracommC2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9IntracommC2Ev,@function
_ZN3MPI9IntracommC2Ev:                  # @_ZN3MPI9IntracommC2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI4CommC2Ev@PLT
	movq	-16(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI9IntracommE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end221:
	.size	_ZN3MPI9IntracommC2Ev, .Lfunc_end221-_ZN3MPI9IntracommC2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI14Is_initializedEv,"axG",@progbits,_ZN3MPI14Is_initializedEv,comdat
	.weak	_ZN3MPI14Is_initializedEv       # -- Begin function _ZN3MPI14Is_initializedEv
	.p2align	4, 0x90
	.type	_ZN3MPI14Is_initializedEv,@function
_ZN3MPI14Is_initializedEv:              # @_ZN3MPI14Is_initializedEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	leaq	-4(%rbp), %rdi
	callq	MPI_Initialized@PLT
	cmpl	$0, -4(%rbp)
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end222:
	.size	_ZN3MPI14Is_initializedEv, .Lfunc_end222-_ZN3MPI14Is_initializedEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4CommC2EP19ompi_communicator_t,"axG",@progbits,_ZN3MPI4CommC2EP19ompi_communicator_t,comdat
	.weak	_ZN3MPI4CommC2EP19ompi_communicator_t # -- Begin function _ZN3MPI4CommC2EP19ompi_communicator_t
	.p2align	4, 0x90
	.type	_ZN3MPI4CommC2EP19ompi_communicator_t,@function
_ZN3MPI4CommC2EP19ompi_communicator_t:  # @_ZN3MPI4CommC2EP19ompi_communicator_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -24(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rsi
	callq	_ZN3MPI9Comm_NullC2EP19ompi_communicator_t
	movq	-24(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI4CommE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end223:
	.size	_ZN3MPI4CommC2EP19ompi_communicator_t, .Lfunc_end223-_ZN3MPI4CommC2EP19ompi_communicator_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9Comm_NullC2EP19ompi_communicator_t,"axG",@progbits,_ZN3MPI9Comm_NullC2EP19ompi_communicator_t,comdat
	.weak	_ZN3MPI9Comm_NullC2EP19ompi_communicator_t # -- Begin function _ZN3MPI9Comm_NullC2EP19ompi_communicator_t
	.p2align	4, 0x90
	.type	_ZN3MPI9Comm_NullC2EP19ompi_communicator_t,@function
_ZN3MPI9Comm_NullC2EP19ompi_communicator_t: # @_ZN3MPI9Comm_NullC2EP19ompi_communicator_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	_ZTVN3MPI9Comm_NullE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end224:
	.size	_ZN3MPI9Comm_NullC2EP19ompi_communicator_t, .Lfunc_end224-_ZN3MPI9Comm_NullC2EP19ompi_communicator_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9Comm_NullD2Ev,"axG",@progbits,_ZN3MPI9Comm_NullD2Ev,comdat
	.weak	_ZN3MPI9Comm_NullD2Ev           # -- Begin function _ZN3MPI9Comm_NullD2Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9Comm_NullD2Ev,@function
_ZN3MPI9Comm_NullD2Ev:                  # @_ZN3MPI9Comm_NullD2Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end225:
	.size	_ZN3MPI9Comm_NullD2Ev, .Lfunc_end225-_ZN3MPI9Comm_NullD2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9Comm_NullD0Ev,"axG",@progbits,_ZN3MPI9Comm_NullD0Ev,comdat
	.weak	_ZN3MPI9Comm_NullD0Ev           # -- Begin function _ZN3MPI9Comm_NullD0Ev
	.p2align	4, 0x90
	.type	_ZN3MPI9Comm_NullD0Ev,@function
_ZN3MPI9Comm_NullD0Ev:                  # @_ZN3MPI9Comm_NullD0Ev
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -16(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI9Comm_NullD2Ev
	movq	-16(%rbp), %rdi                 # 8-byte Reload
	callq	_ZdlPv@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end226:
	.size	_ZN3MPI9Comm_NullD0Ev, .Lfunc_end226-_ZN3MPI9Comm_NullD0Ev
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv,"axG",@progbits,_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv,comdat
	.weak	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv # -- Begin function _ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	.p2align	4, 0x90
	.type	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv,@function
_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv: # @_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end227:
	.size	_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv, .Lfunc_end227-_ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8DatatypeaSERKP15ompi_datatype_t,"axG",@progbits,_ZN3MPI8DatatypeaSERKP15ompi_datatype_t,comdat
	.weak	_ZN3MPI8DatatypeaSERKP15ompi_datatype_t # -- Begin function _ZN3MPI8DatatypeaSERKP15ompi_datatype_t
	.p2align	4, 0x90
	.type	_ZN3MPI8DatatypeaSERKP15ompi_datatype_t,@function
_ZN3MPI8DatatypeaSERKP15ompi_datatype_t: # @_ZN3MPI8DatatypeaSERKP15ompi_datatype_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end228:
	.size	_ZN3MPI8DatatypeaSERKP15ompi_datatype_t, .Lfunc_end228-_ZN3MPI8DatatypeaSERKP15ompi_datatype_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI6StatusaSERK20ompi_status_public_t,"axG",@progbits,_ZN3MPI6StatusaSERK20ompi_status_public_t,comdat
	.weak	_ZN3MPI6StatusaSERK20ompi_status_public_t # -- Begin function _ZN3MPI6StatusaSERK20ompi_status_public_t
	.p2align	4, 0x90
	.type	_ZN3MPI6StatusaSERK20ompi_status_public_t,@function
_ZN3MPI6StatusaSERK20ompi_status_public_t: # @_ZN3MPI6StatusaSERK20ompi_status_public_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movq	(%rcx), %rdx
	movq	%rdx, 8(%rax)
	movq	8(%rcx), %rdx
	movq	%rdx, 16(%rax)
	movq	16(%rcx), %rcx
	movq	%rcx, 24(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end229:
	.size	_ZN3MPI6StatusaSERK20ompi_status_public_t, .Lfunc_end229-_ZN3MPI6StatusaSERK20ompi_status_public_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI5GroupC2EP12ompi_group_t,"axG",@progbits,_ZN3MPI5GroupC2EP12ompi_group_t,comdat
	.weak	_ZN3MPI5GroupC2EP12ompi_group_t # -- Begin function _ZN3MPI5GroupC2EP12ompi_group_t
	.p2align	4, 0x90
	.type	_ZN3MPI5GroupC2EP12ompi_group_t,@function
_ZN3MPI5GroupC2EP12ompi_group_t:        # @_ZN3MPI5GroupC2EP12ompi_group_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	_ZTVN3MPI5GroupE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end230:
	.size	_ZN3MPI5GroupC2EP12ompi_group_t, .Lfunc_end230-_ZN3MPI5GroupC2EP12ompi_group_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI7RequestC2EP14ompi_request_t,"axG",@progbits,_ZN3MPI7RequestC2EP14ompi_request_t,comdat
	.weak	_ZN3MPI7RequestC2EP14ompi_request_t # -- Begin function _ZN3MPI7RequestC2EP14ompi_request_t
	.p2align	4, 0x90
	.type	_ZN3MPI7RequestC2EP14ompi_request_t,@function
_ZN3MPI7RequestC2EP14ompi_request_t:    # @_ZN3MPI7RequestC2EP14ompi_request_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	_ZTVN3MPI7RequestE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end231:
	.size	_ZN3MPI7RequestC2EP14ompi_request_t, .Lfunc_end231-_ZN3MPI7RequestC2EP14ompi_request_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI8PrequestC2ERKP14ompi_request_t,"axG",@progbits,_ZN3MPI8PrequestC2ERKP14ompi_request_t,comdat
	.weak	_ZN3MPI8PrequestC2ERKP14ompi_request_t # -- Begin function _ZN3MPI8PrequestC2ERKP14ompi_request_t
	.p2align	4, 0x90
	.type	_ZN3MPI8PrequestC2ERKP14ompi_request_t,@function
_ZN3MPI8PrequestC2ERKP14ompi_request_t: # @_ZN3MPI8PrequestC2ERKP14ompi_request_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -24(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rax
	movq	(%rax), %rsi
	callq	_ZN3MPI7RequestC2EP14ompi_request_t
	movq	-24(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI8PrequestE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end232:
	.size	_ZN3MPI8PrequestC2ERKP14ompi_request_t, .Lfunc_end232-_ZN3MPI8PrequestC2ERKP14ompi_request_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t,"axG",@progbits,_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t,comdat
	.weak	_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t # -- Begin function _ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t
	.p2align	4, 0x90
	.type	_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t,@function
_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t: # @_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	_ZTVN3MPI10ErrhandlerE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end233:
	.size	_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t, .Lfunc_end233-_ZN3MPI10ErrhandlerC2EP17ompi_errhandler_t
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNK3MPI5GroupcvP12ompi_group_tEv,"axG",@progbits,_ZNK3MPI5GroupcvP12ompi_group_tEv,comdat
	.weak	_ZNK3MPI5GroupcvP12ompi_group_tEv # -- Begin function _ZNK3MPI5GroupcvP12ompi_group_tEv
	.p2align	4, 0x90
	.type	_ZNK3MPI5GroupcvP12ompi_group_tEv,@function
_ZNK3MPI5GroupcvP12ompi_group_tEv:      # @_ZNK3MPI5GroupcvP12ompi_group_tEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end234:
	.size	_ZNK3MPI5GroupcvP12ompi_group_tEv, .Lfunc_end234-_ZNK3MPI5GroupcvP12ompi_group_tEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9IntracommC2EP19ompi_communicator_t,"axG",@progbits,_ZN3MPI9IntracommC2EP19ompi_communicator_t,comdat
	.weak	_ZN3MPI9IntracommC2EP19ompi_communicator_t # -- Begin function _ZN3MPI9IntracommC2EP19ompi_communicator_t
	.p2align	4, 0x90
	.type	_ZN3MPI9IntracommC2EP19ompi_communicator_t,@function
_ZN3MPI9IntracommC2EP19ompi_communicator_t: # @_ZN3MPI9IntracommC2EP19ompi_communicator_t
.Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception6
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	%rdi, -48(%rbp)                 # 8-byte Spill
	callq	_ZN3MPI4CommC2Ev@PLT
	movq	-48(%rbp), %rax                 # 8-byte Reload
	leaq	_ZTVN3MPI9IntracommE+16(%rip), %rcx
	movq	%rcx, (%rax)
	movl	$0, -20(%rbp)
.Ltmp22:
	callq	_ZN3MPI14Is_initializedEv
.Ltmp23:
	movb	%al, -37(%rbp)                  # 1-byte Spill
	jmp	.LBB235_1
.LBB235_1:
	movb	-37(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB235_2
	jmp	.LBB235_9
.LBB235_2:
	movq	ompi_mpi_comm_null@GOTPCREL(%rip), %rax
	cmpq	%rax, -16(%rbp)
	je	.LBB235_9
# %bb.3:
	movq	-16(%rbp), %rdi
.Ltmp24:
	leaq	-20(%rbp), %rsi
	callq	MPI_Comm_test_inter@PLT
.Ltmp25:
	jmp	.LBB235_4
.LBB235_4:
	cmpl	$0, -20(%rbp)
	je	.LBB235_7
# %bb.5:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	ompi_mpi_comm_null@GOTPCREL(%rip), %rcx
	movq	%rcx, 8(%rax)
	jmp	.LBB235_8
.LBB235_6:
.Ltmp26:
	movq	-48(%rbp), %rdi                 # 8-byte Reload
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -32(%rbp)
	movl	%eax, -36(%rbp)
	callq	_ZN3MPI4CommD2Ev
	jmp	.LBB235_11
.LBB235_7:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
.LBB235_8:
	jmp	.LBB235_10
.LBB235_9:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
.LBB235_10:
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB235_11:
	.cfi_def_cfa %rbp, 16
	movq	-32(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end235:
	.size	_ZN3MPI9IntracommC2EP19ompi_communicator_t, .Lfunc_end235-_ZN3MPI9IntracommC2EP19ompi_communicator_t
	.cfi_endproc
	.section	.gcc_except_table._ZN3MPI9IntracommC2EP19ompi_communicator_t,"aG",@progbits,_ZN3MPI9IntracommC2EP19ompi_communicator_t,comdat
	.p2align	2
GCC_except_table235:
.Lexception6:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end6-.Lcst_begin6
.Lcst_begin6:
	.uleb128 .Lfunc_begin6-.Lfunc_begin6    # >> Call Site 1 <<
	.uleb128 .Ltmp22-.Lfunc_begin6          #   Call between .Lfunc_begin6 and .Ltmp22
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp22-.Lfunc_begin6          # >> Call Site 2 <<
	.uleb128 .Ltmp25-.Ltmp22                #   Call between .Ltmp22 and .Ltmp25
	.uleb128 .Ltmp26-.Lfunc_begin6          #     jumps to .Ltmp26
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp25-.Lfunc_begin6          # >> Call Site 3 <<
	.uleb128 .Lfunc_end235-.Ltmp25          #   Call between .Ltmp25 and .Lfunc_end235
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end6:
	.p2align	2
                                        # -- End function
	.section	.text._ZNK3MPI4InfocvP11ompi_info_tEv,"axG",@progbits,_ZNK3MPI4InfocvP11ompi_info_tEv,comdat
	.weak	_ZNK3MPI4InfocvP11ompi_info_tEv # -- Begin function _ZNK3MPI4InfocvP11ompi_info_tEv
	.p2align	4, 0x90
	.type	_ZNK3MPI4InfocvP11ompi_info_tEv,@function
_ZNK3MPI4InfocvP11ompi_info_tEv:        # @_ZNK3MPI4InfocvP11ompi_info_tEv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end236:
	.size	_ZNK3MPI4InfocvP11ompi_info_tEv, .Lfunc_end236-_ZNK3MPI4InfocvP11ompi_info_tEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE,"axG",@progbits,_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE,comdat
	.weak	_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE # -- Begin function _ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE
	.p2align	4, 0x90
	.type	_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE,@function
_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE: # @_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	%edi, -4(%rbp)
	movq	%rsi, -16(%rbp)
	movslq	-4(%rbp), %rax
	movl	$8, %ecx
	mulq	%rcx
	movq	%rax, %rdi
	seto	%al
	movq	$-1, %rax
	cmovoq	%rax, %rdi
	callq	_Znam@PLT
	movq	%rax, -24(%rbp)
	movl	$0, -28(%rbp)
.LBB237_1:                              # =>This Inner Loop Header: Depth=1
	movl	-28(%rbp), %eax
	cmpl	-4(%rbp), %eax
	jge	.LBB237_4
# %bb.2:                                #   in Loop: Header=BB237_1 Depth=1
	movq	-16(%rbp), %rdi
	movslq	-28(%rbp), %rax
	shlq	$4, %rax
	addq	%rax, %rdi
	callq	_ZNK3MPI4InfocvP11ompi_info_tEv
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	movslq	-28(%rbp), %rcx
	movq	%rdx, (%rax,%rcx,8)
# %bb.3:                                #   in Loop: Header=BB237_1 Depth=1
	movl	-28(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -28(%rbp)
	jmp	.LBB237_1
.LBB237_4:
	movq	-24(%rbp), %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end237:
	.size	_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE, .Lfunc_end237-_ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN3MPI4InfoC2EP11ompi_info_t,"axG",@progbits,_ZN3MPI4InfoC2EP11ompi_info_t,comdat
	.weak	_ZN3MPI4InfoC2EP11ompi_info_t   # -- Begin function _ZN3MPI4InfoC2EP11ompi_info_t
	.p2align	4, 0x90
	.type	_ZN3MPI4InfoC2EP11ompi_info_t,@function
_ZN3MPI4InfoC2EP11ompi_info_t:          # @_ZN3MPI4InfoC2EP11ompi_info_t
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	_ZTVN3MPI4InfoE(%rip), %rcx
	addq	$16, %rcx
	movq	%rcx, (%rax)
	movq	-16(%rbp), %rcx
	movq	%rcx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end238:
	.size	_ZN3MPI4InfoC2EP11ompi_info_t, .Lfunc_end238-_ZN3MPI4InfoC2EP11ompi_info_t
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_simple_nn.cu
	.type	_GLOBAL__sub_I_simple_nn.cu,@function
_GLOBAL__sub_I_simple_nn.cu:            # @_GLOBAL__sub_I_simple_nn.cu
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	__cxx_global_var_init
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end239:
	.size	_GLOBAL__sub_I_simple_nn.cu, .Lfunc_end239-_GLOBAL__sub_I_simple_nn.cu
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function __cuda_register_globals
	.type	__cuda_register_globals,@function
__cuda_register_globals:                # @__cuda_register_globals
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	%rdi, 32(%rsp)                  # 8-byte Spill
	leaq	_Z21__device_stub__vecAddPfS_S_i(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movl	$4294967295, %r8d               # imm = 0xFFFFFFFF
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%rcx, %rdx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	movq	$0, 16(%rsp)
	movq	$0, 24(%rsp)
	callq	__cudaRegisterFunction@PLT
	movq	32(%rsp), %rdi                  # 8-byte Reload
	leaq	_Z27__device_stub__forward_passPfS_S_ii(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movl	$4294967295, %r8d               # imm = 0xFFFFFFFF
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%rcx, %rdx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	movq	$0, 16(%rsp)
	movq	$0, 24(%rsp)
	callq	__cudaRegisterFunction@PLT
	movq	32(%rsp), %rdi                  # 8-byte Reload
	leaq	_Z28__device_stub__backward_passPfS_S_i(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movl	$4294967295, %r8d               # imm = 0xFFFFFFFF
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%rcx, %rdx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	movq	$0, 16(%rsp)
	movq	$0, 24(%rsp)
	callq	__cudaRegisterFunction@PLT
	movq	32(%rsp), %rdi                  # 8-byte Reload
	leaq	_Z25__device_stub__calc_gradsPfS_S_ii(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movl	$4294967295, %r8d               # imm = 0xFFFFFFFF
	xorl	%eax, %eax
	movl	%eax, %r9d
	movq	%rcx, %rdx
	movq	$0, (%rsp)
	movq	$0, 8(%rsp)
	movq	$0, 16(%rsp)
	movq	$0, 24(%rsp)
	callq	__cudaRegisterFunction@PLT
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end240:
	.size	__cuda_register_globals, .Lfunc_end240-__cuda_register_globals
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __cuda_module_ctor
	.type	__cuda_module_ctor,@function
__cuda_module_ctor:                     # @__cuda_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	leaq	__cuda_fatbin_wrapper(%rip), %rdi
	callq	__cudaRegisterFatBinary@PLT
	movq	%rax, %rdi
	movq	%rdi, (%rsp)                    # 8-byte Spill
	movq	%rdi, __cuda_gpubin_handle(%rip)
	callq	__cuda_register_globals
	movq	(%rsp), %rdi                    # 8-byte Reload
	callq	__cudaRegisterFatBinaryEnd@PLT
	leaq	__cuda_module_dtor(%rip), %rdi
	callq	atexit@PLT
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end241:
	.size	__cuda_module_ctor, .Lfunc_end241-__cuda_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __cuda_module_dtor
	.type	__cuda_module_dtor,@function
__cuda_module_dtor:                     # @__cuda_module_dtor
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	__cuda_gpubin_handle(%rip), %rdi
	callq	__cudaUnregisterFatBinary@PLT
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end242:
	.size	__cuda_module_dtor, .Lfunc_end242-__cuda_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Failed: MPI error %s:%d '%d'\n"
	.size	.L.str, 30

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"simple_nn.cu"
	.size	.L.str.1, 13

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Failed: Cuda error %s:%d '%s'\n"
	.size	.L.str.2, 31

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"Failed, NCCL error %s:%d '%s'\n"
	.size	.L.str.3, 31

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"[MPI Rank %d] Before AllReduce: ---> Gradient content: First = %.2f, Mid = %.2f, Last = %.2f \n"
	.size	.L.str.4, 95

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"[MPI Rank %d]  After AllReduce: ---> Gradient content: First = %.2f, Mid = %.2f, Last = %.2f \n"
	.size	.L.str.5, 95

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"[MPI Rank %d] Finalized\n"
	.size	.L.str.6, 25

	.type	_ZTVN3MPI8DatatypeE,@object     # @_ZTVN3MPI8DatatypeE
	.section	.data.rel.ro._ZTVN3MPI8DatatypeE,"aGw",@progbits,_ZTVN3MPI8DatatypeE,comdat
	.weak	_ZTVN3MPI8DatatypeE
	.p2align	3
_ZTVN3MPI8DatatypeE:
	.quad	0
	.quad	_ZTIN3MPI8DatatypeE
	.quad	_ZN3MPI8DatatypeD2Ev
	.quad	_ZN3MPI8DatatypeD0Ev
	.quad	_ZNK3MPI8Datatype17Create_contiguousEi
	.quad	_ZNK3MPI8Datatype13Create_vectorEiii
	.quad	_ZNK3MPI8Datatype14Create_indexedEiPKiS2_
	.quad	_ZNK3MPI8Datatype15Create_hindexedEiPKiPKl
	.quad	_ZNK3MPI8Datatype14Create_hvectorEiil
	.quad	_ZNK3MPI8Datatype20Create_indexed_blockEiiPKi
	.quad	_ZNK3MPI8Datatype14Create_resizedEll
	.quad	_ZNK3MPI8Datatype8Get_sizeEv
	.quad	_ZNK3MPI8Datatype10Get_extentERlS1_
	.quad	_ZNK3MPI8Datatype15Get_true_extentERlS1_
	.quad	_ZN3MPI8Datatype6CommitEv
	.quad	_ZN3MPI8Datatype4FreeEv
	.quad	_ZNK3MPI8Datatype4PackEPKviPviRiRKNS_4CommE
	.quad	_ZNK3MPI8Datatype6UnpackEPKviPviRiRKNS_4CommE
	.quad	_ZNK3MPI8Datatype9Pack_sizeEiRKNS_4CommE
	.quad	_ZNK3MPI8Datatype13Pack_externalEPKcPKviPvlRl
	.quad	_ZNK3MPI8Datatype18Pack_external_sizeEPKci
	.quad	_ZNK3MPI8Datatype15Unpack_externalEPKcPKvlRlPvi
	.quad	_ZNK3MPI8Datatype15Create_subarrayEiPKiS2_S2_i
	.quad	_ZNK3MPI8Datatype13Create_darrayEiiiPKiS2_S2_S2_i
	.quad	_ZNK3MPI8Datatype3DupEv
	.quad	_ZN3MPI8Datatype11Delete_attrEi
	.quad	_ZNK3MPI8Datatype8Get_attrEiPv
	.quad	_ZNK3MPI8Datatype12Get_contentsEiiiPiPlPS0_
	.quad	_ZNK3MPI8Datatype12Get_envelopeERiS1_S1_S1_
	.quad	_ZNK3MPI8Datatype8Get_nameEPcRi
	.quad	_ZN3MPI8Datatype8Set_attrEiPKv
	.quad	_ZN3MPI8Datatype8Set_nameEPKc
	.size	_ZTVN3MPI8DatatypeE, 256

	.type	_ZTSN3MPI8DatatypeE,@object     # @_ZTSN3MPI8DatatypeE
	.section	.rodata._ZTSN3MPI8DatatypeE,"aG",@progbits,_ZTSN3MPI8DatatypeE,comdat
	.weak	_ZTSN3MPI8DatatypeE
_ZTSN3MPI8DatatypeE:
	.asciz	"N3MPI8DatatypeE"
	.size	_ZTSN3MPI8DatatypeE, 16

	.type	_ZTIN3MPI8DatatypeE,@object     # @_ZTIN3MPI8DatatypeE
	.section	.data.rel.ro._ZTIN3MPI8DatatypeE,"aGw",@progbits,_ZTIN3MPI8DatatypeE,comdat
	.weak	_ZTIN3MPI8DatatypeE
	.p2align	3
_ZTIN3MPI8DatatypeE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI8DatatypeE
	.size	_ZTIN3MPI8DatatypeE, 16

	.type	_ZTVN3MPI6StatusE,@object       # @_ZTVN3MPI6StatusE
	.section	.data.rel.ro._ZTVN3MPI6StatusE,"aGw",@progbits,_ZTVN3MPI6StatusE,comdat
	.weak	_ZTVN3MPI6StatusE
	.p2align	3
_ZTVN3MPI6StatusE:
	.quad	0
	.quad	_ZTIN3MPI6StatusE
	.quad	_ZN3MPI6StatusD2Ev
	.quad	_ZN3MPI6StatusD0Ev
	.quad	_ZNK3MPI6Status9Get_countERKNS_8DatatypeE
	.quad	_ZNK3MPI6Status12Is_cancelledEv
	.quad	_ZNK3MPI6Status12Get_elementsERKNS_8DatatypeE
	.quad	_ZNK3MPI6Status10Get_sourceEv
	.quad	_ZN3MPI6Status10Set_sourceEi
	.quad	_ZNK3MPI6Status7Get_tagEv
	.quad	_ZN3MPI6Status7Set_tagEi
	.quad	_ZNK3MPI6Status9Get_errorEv
	.quad	_ZN3MPI6Status9Set_errorEi
	.quad	_ZN3MPI6Status12Set_elementsERKNS_8DatatypeEi
	.quad	_ZN3MPI6Status13Set_cancelledEb
	.size	_ZTVN3MPI6StatusE, 120

	.type	_ZTSN3MPI6StatusE,@object       # @_ZTSN3MPI6StatusE
	.section	.rodata._ZTSN3MPI6StatusE,"aG",@progbits,_ZTSN3MPI6StatusE,comdat
	.weak	_ZTSN3MPI6StatusE
_ZTSN3MPI6StatusE:
	.asciz	"N3MPI6StatusE"
	.size	_ZTSN3MPI6StatusE, 14

	.type	_ZTIN3MPI6StatusE,@object       # @_ZTIN3MPI6StatusE
	.section	.data.rel.ro._ZTIN3MPI6StatusE,"aGw",@progbits,_ZTIN3MPI6StatusE,comdat
	.weak	_ZTIN3MPI6StatusE
	.p2align	3
_ZTIN3MPI6StatusE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI6StatusE
	.size	_ZTIN3MPI6StatusE, 16

	.type	_ZTVN3MPI7RequestE,@object      # @_ZTVN3MPI7RequestE
	.section	.data.rel.ro._ZTVN3MPI7RequestE,"aGw",@progbits,_ZTVN3MPI7RequestE,comdat
	.weak	_ZTVN3MPI7RequestE
	.p2align	3
_ZTVN3MPI7RequestE:
	.quad	0
	.quad	_ZTIN3MPI7RequestE
	.quad	_ZN3MPI7RequestD2Ev
	.quad	_ZN3MPI7RequestD0Ev
	.quad	_ZN3MPI7Request4WaitERNS_6StatusE
	.quad	_ZN3MPI7Request4WaitEv
	.quad	_ZN3MPI7Request4TestERNS_6StatusE
	.quad	_ZN3MPI7Request4TestEv
	.quad	_ZN3MPI7Request4FreeEv
	.quad	_ZNK3MPI7Request6CancelEv
	.quad	_ZNK3MPI7Request10Get_statusERNS_6StatusE
	.quad	_ZNK3MPI7Request10Get_statusEv
	.size	_ZTVN3MPI7RequestE, 96

	.type	_ZTSN3MPI7RequestE,@object      # @_ZTSN3MPI7RequestE
	.section	.rodata._ZTSN3MPI7RequestE,"aG",@progbits,_ZTSN3MPI7RequestE,comdat
	.weak	_ZTSN3MPI7RequestE
_ZTSN3MPI7RequestE:
	.asciz	"N3MPI7RequestE"
	.size	_ZTSN3MPI7RequestE, 15

	.type	_ZTIN3MPI7RequestE,@object      # @_ZTIN3MPI7RequestE
	.section	.data.rel.ro._ZTIN3MPI7RequestE,"aGw",@progbits,_ZTIN3MPI7RequestE,comdat
	.weak	_ZTIN3MPI7RequestE
	.p2align	3
_ZTIN3MPI7RequestE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI7RequestE
	.size	_ZTIN3MPI7RequestE, 16

	.type	_ZTVN3MPI8PrequestE,@object     # @_ZTVN3MPI8PrequestE
	.section	.data.rel.ro._ZTVN3MPI8PrequestE,"aGw",@progbits,_ZTVN3MPI8PrequestE,comdat
	.weak	_ZTVN3MPI8PrequestE
	.p2align	3
_ZTVN3MPI8PrequestE:
	.quad	0
	.quad	_ZTIN3MPI8PrequestE
	.quad	_ZN3MPI8PrequestD2Ev
	.quad	_ZN3MPI8PrequestD0Ev
	.quad	_ZN3MPI7Request4WaitERNS_6StatusE
	.quad	_ZN3MPI7Request4WaitEv
	.quad	_ZN3MPI7Request4TestERNS_6StatusE
	.quad	_ZN3MPI7Request4TestEv
	.quad	_ZN3MPI7Request4FreeEv
	.quad	_ZNK3MPI7Request6CancelEv
	.quad	_ZNK3MPI7Request10Get_statusERNS_6StatusE
	.quad	_ZNK3MPI7Request10Get_statusEv
	.quad	_ZN3MPI8Prequest5StartEv
	.size	_ZTVN3MPI8PrequestE, 104

	.type	_ZTSN3MPI8PrequestE,@object     # @_ZTSN3MPI8PrequestE
	.section	.rodata._ZTSN3MPI8PrequestE,"aG",@progbits,_ZTSN3MPI8PrequestE,comdat
	.weak	_ZTSN3MPI8PrequestE
_ZTSN3MPI8PrequestE:
	.asciz	"N3MPI8PrequestE"
	.size	_ZTSN3MPI8PrequestE, 16

	.type	_ZTIN3MPI8PrequestE,@object     # @_ZTIN3MPI8PrequestE
	.section	.data.rel.ro._ZTIN3MPI8PrequestE,"aGw",@progbits,_ZTIN3MPI8PrequestE,comdat
	.weak	_ZTIN3MPI8PrequestE
	.p2align	3
_ZTIN3MPI8PrequestE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI8PrequestE
	.quad	_ZTIN3MPI7RequestE
	.size	_ZTIN3MPI8PrequestE, 24

	.type	_ZTVN3MPI8GrequestE,@object     # @_ZTVN3MPI8GrequestE
	.section	.data.rel.ro._ZTVN3MPI8GrequestE,"aGw",@progbits,_ZTVN3MPI8GrequestE,comdat
	.weak	_ZTVN3MPI8GrequestE
	.p2align	3
_ZTVN3MPI8GrequestE:
	.quad	0
	.quad	_ZTIN3MPI8GrequestE
	.quad	_ZN3MPI8GrequestD2Ev
	.quad	_ZN3MPI8GrequestD0Ev
	.quad	_ZN3MPI7Request4WaitERNS_6StatusE
	.quad	_ZN3MPI7Request4WaitEv
	.quad	_ZN3MPI7Request4TestERNS_6StatusE
	.quad	_ZN3MPI7Request4TestEv
	.quad	_ZN3MPI7Request4FreeEv
	.quad	_ZNK3MPI7Request6CancelEv
	.quad	_ZNK3MPI7Request10Get_statusERNS_6StatusE
	.quad	_ZNK3MPI7Request10Get_statusEv
	.quad	_ZN3MPI8Grequest8CompleteEv
	.size	_ZTVN3MPI8GrequestE, 104

	.type	_ZTSN3MPI8GrequestE,@object     # @_ZTSN3MPI8GrequestE
	.section	.rodata._ZTSN3MPI8GrequestE,"aG",@progbits,_ZTSN3MPI8GrequestE,comdat
	.weak	_ZTSN3MPI8GrequestE
_ZTSN3MPI8GrequestE:
	.asciz	"N3MPI8GrequestE"
	.size	_ZTSN3MPI8GrequestE, 16

	.type	_ZTIN3MPI8GrequestE,@object     # @_ZTIN3MPI8GrequestE
	.section	.data.rel.ro._ZTIN3MPI8GrequestE,"aGw",@progbits,_ZTIN3MPI8GrequestE,comdat
	.weak	_ZTIN3MPI8GrequestE
	.p2align	3
_ZTIN3MPI8GrequestE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI8GrequestE
	.quad	_ZTIN3MPI7RequestE
	.size	_ZTIN3MPI8GrequestE, 24

	.type	_ZTVN3MPI5GroupE,@object        # @_ZTVN3MPI5GroupE
	.section	.data.rel.ro._ZTVN3MPI5GroupE,"aGw",@progbits,_ZTVN3MPI5GroupE,comdat
	.weak	_ZTVN3MPI5GroupE
	.p2align	3
_ZTVN3MPI5GroupE:
	.quad	0
	.quad	_ZTIN3MPI5GroupE
	.quad	_ZN3MPI5GroupD2Ev
	.quad	_ZN3MPI5GroupD0Ev
	.quad	_ZNK3MPI5Group8Get_sizeEv
	.quad	_ZNK3MPI5Group8Get_rankEv
	.quad	_ZNK3MPI5Group4InclEiPKi
	.quad	_ZNK3MPI5Group4ExclEiPKi
	.quad	_ZNK3MPI5Group10Range_inclEiPA3_Ki
	.quad	_ZNK3MPI5Group10Range_exclEiPA3_Ki
	.quad	_ZN3MPI5Group4FreeEv
	.size	_ZTVN3MPI5GroupE, 88

	.type	_ZTSN3MPI5GroupE,@object        # @_ZTSN3MPI5GroupE
	.section	.rodata._ZTSN3MPI5GroupE,"aG",@progbits,_ZTSN3MPI5GroupE,comdat
	.weak	_ZTSN3MPI5GroupE
_ZTSN3MPI5GroupE:
	.asciz	"N3MPI5GroupE"
	.size	_ZTSN3MPI5GroupE, 13

	.type	_ZTIN3MPI5GroupE,@object        # @_ZTIN3MPI5GroupE
	.section	.data.rel.ro._ZTIN3MPI5GroupE,"aGw",@progbits,_ZTIN3MPI5GroupE,comdat
	.weak	_ZTIN3MPI5GroupE
	.p2align	3
_ZTIN3MPI5GroupE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI5GroupE
	.size	_ZTIN3MPI5GroupE, 16

	.type	_ZTVN3MPI4CommE,@object         # @_ZTVN3MPI4CommE
	.section	.data.rel.ro._ZTVN3MPI4CommE,"aGw",@progbits,_ZTVN3MPI4CommE,comdat
	.weak	_ZTVN3MPI4CommE
	.p2align	3
_ZTVN3MPI4CommE:
	.quad	0
	.quad	_ZTIN3MPI4CommE
	.quad	_ZN3MPI4CommD2Ev
	.quad	_ZN3MPI4CommD0Ev
	.quad	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm6IprobeEii
	.quad	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm5ProbeEii
	.quad	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.quad	_ZNK3MPI4Comm9Get_groupEv
	.quad	_ZNK3MPI4Comm8Get_sizeEv
	.quad	_ZNK3MPI4Comm8Get_rankEv
	.quad	__cxa_pure_virtual
	.quad	_ZN3MPI4Comm4FreeEv
	.quad	_ZNK3MPI4Comm8Is_interEv
	.quad	_ZNK3MPI4Comm7BarrierEv
	.quad	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.quad	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.quad	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.quad	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.quad	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.quad	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.quad	_ZN3MPI4Comm10DisconnectEv
	.quad	_ZNK3MPI4Comm8Get_nameEPcRi
	.quad	_ZN3MPI4Comm8Set_nameEPKc
	.quad	_ZNK3MPI4Comm12Get_topologyEv
	.quad	_ZN3MPI4Comm5AbortEi
	.quad	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.quad	_ZNK3MPI4Comm14Get_errhandlerEv
	.quad	_ZNK3MPI4Comm8Set_attrEiPKv
	.quad	_ZNK3MPI4Comm8Get_attrEiPv
	.quad	_ZN3MPI4Comm11Delete_attrEi
	.size	_ZTVN3MPI4CommE, 464

	.type	_ZTSN3MPI4CommE,@object         # @_ZTSN3MPI4CommE
	.section	.rodata._ZTSN3MPI4CommE,"aG",@progbits,_ZTSN3MPI4CommE,comdat
	.weak	_ZTSN3MPI4CommE
_ZTSN3MPI4CommE:
	.asciz	"N3MPI4CommE"
	.size	_ZTSN3MPI4CommE, 12

	.type	_ZTSN3MPI9Comm_NullE,@object    # @_ZTSN3MPI9Comm_NullE
	.section	.rodata._ZTSN3MPI9Comm_NullE,"aG",@progbits,_ZTSN3MPI9Comm_NullE,comdat
	.weak	_ZTSN3MPI9Comm_NullE
_ZTSN3MPI9Comm_NullE:
	.asciz	"N3MPI9Comm_NullE"
	.size	_ZTSN3MPI9Comm_NullE, 17

	.type	_ZTIN3MPI9Comm_NullE,@object    # @_ZTIN3MPI9Comm_NullE
	.section	.data.rel.ro._ZTIN3MPI9Comm_NullE,"aGw",@progbits,_ZTIN3MPI9Comm_NullE,comdat
	.weak	_ZTIN3MPI9Comm_NullE
	.p2align	3
_ZTIN3MPI9Comm_NullE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI9Comm_NullE
	.size	_ZTIN3MPI9Comm_NullE, 16

	.type	_ZTIN3MPI4CommE,@object         # @_ZTIN3MPI4CommE
	.section	.data.rel.ro._ZTIN3MPI4CommE,"aGw",@progbits,_ZTIN3MPI4CommE,comdat
	.weak	_ZTIN3MPI4CommE
	.p2align	3
_ZTIN3MPI4CommE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI4CommE
	.quad	_ZTIN3MPI9Comm_NullE
	.size	_ZTIN3MPI4CommE, 24

	.type	_ZTVN3MPI3WinE,@object          # @_ZTVN3MPI3WinE
	.section	.data.rel.ro._ZTVN3MPI3WinE,"aGw",@progbits,_ZTVN3MPI3WinE,comdat
	.weak	_ZTVN3MPI3WinE
	.p2align	3
_ZTVN3MPI3WinE:
	.quad	0
	.quad	_ZTIN3MPI3WinE
	.quad	_ZN3MPI3WinD2Ev
	.quad	_ZN3MPI3WinD0Ev
	.quad	_ZNK3MPI3Win14Set_errhandlerERKNS_10ErrhandlerE
	.quad	_ZNK3MPI3Win14Get_errhandlerEv
	.quad	_ZNK3MPI3Win10AccumulateEPKviRKNS_8DatatypeEiliS5_RKNS_2OpE
	.quad	_ZNK3MPI3Win8CompleteEv
	.quad	_ZNK3MPI3Win5FenceEi
	.quad	_ZN3MPI3Win4FreeEv
	.quad	_ZNK3MPI3Win3GetEPKviRKNS_8DatatypeEiliS5_
	.quad	_ZNK3MPI3Win9Get_groupEv
	.quad	_ZNK3MPI3Win4LockEiii
	.quad	_ZNK3MPI3Win4PostERKNS_5GroupEi
	.quad	_ZNK3MPI3Win3PutEPKviRKNS_8DatatypeEiliS5_
	.quad	_ZNK3MPI3Win5StartERKNS_5GroupEi
	.quad	_ZNK3MPI3Win4TestEv
	.quad	_ZNK3MPI3Win6UnlockEi
	.quad	_ZNK3MPI3Win4WaitEv
	.quad	_ZNK3MPI3Win15Call_errhandlerEi
	.quad	_ZN3MPI3Win11Delete_attrEi
	.quad	_ZNK3MPI3Win8Get_nameEPcRi
	.quad	_ZN3MPI3Win8Set_attrEiPKv
	.quad	_ZN3MPI3Win8Set_nameEPKc
	.size	_ZTVN3MPI3WinE, 192

	.type	_ZTSN3MPI3WinE,@object          # @_ZTSN3MPI3WinE
	.section	.rodata._ZTSN3MPI3WinE,"aG",@progbits,_ZTSN3MPI3WinE,comdat
	.weak	_ZTSN3MPI3WinE
_ZTSN3MPI3WinE:
	.asciz	"N3MPI3WinE"
	.size	_ZTSN3MPI3WinE, 11

	.type	_ZTIN3MPI3WinE,@object          # @_ZTIN3MPI3WinE
	.section	.data.rel.ro._ZTIN3MPI3WinE,"aGw",@progbits,_ZTIN3MPI3WinE,comdat
	.weak	_ZTIN3MPI3WinE
	.p2align	3
_ZTIN3MPI3WinE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI3WinE
	.size	_ZTIN3MPI3WinE, 16

	.type	_ZTVN3MPI10ErrhandlerE,@object  # @_ZTVN3MPI10ErrhandlerE
	.section	.data.rel.ro._ZTVN3MPI10ErrhandlerE,"aGw",@progbits,_ZTVN3MPI10ErrhandlerE,comdat
	.weak	_ZTVN3MPI10ErrhandlerE
	.p2align	3
_ZTVN3MPI10ErrhandlerE:
	.quad	0
	.quad	_ZTIN3MPI10ErrhandlerE
	.quad	_ZN3MPI10ErrhandlerD2Ev
	.quad	_ZN3MPI10ErrhandlerD0Ev
	.quad	_ZN3MPI10Errhandler4FreeEv
	.size	_ZTVN3MPI10ErrhandlerE, 40

	.type	_ZTSN3MPI10ErrhandlerE,@object  # @_ZTSN3MPI10ErrhandlerE
	.section	.rodata._ZTSN3MPI10ErrhandlerE,"aG",@progbits,_ZTSN3MPI10ErrhandlerE,comdat
	.weak	_ZTSN3MPI10ErrhandlerE
_ZTSN3MPI10ErrhandlerE:
	.asciz	"N3MPI10ErrhandlerE"
	.size	_ZTSN3MPI10ErrhandlerE, 19

	.type	_ZTIN3MPI10ErrhandlerE,@object  # @_ZTIN3MPI10ErrhandlerE
	.section	.data.rel.ro._ZTIN3MPI10ErrhandlerE,"aGw",@progbits,_ZTIN3MPI10ErrhandlerE,comdat
	.weak	_ZTIN3MPI10ErrhandlerE
	.p2align	3
_ZTIN3MPI10ErrhandlerE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI10ErrhandlerE
	.size	_ZTIN3MPI10ErrhandlerE, 16

	.type	_ZTVN3MPI9IntracommE,@object    # @_ZTVN3MPI9IntracommE
	.section	.data.rel.ro._ZTVN3MPI9IntracommE,"aGw",@progbits,_ZTVN3MPI9IntracommE,comdat
	.weak	_ZTVN3MPI9IntracommE
	.p2align	3
_ZTVN3MPI9IntracommE:
	.quad	0
	.quad	_ZTIN3MPI9IntracommE
	.quad	_ZN3MPI9IntracommD2Ev
	.quad	_ZN3MPI9IntracommD0Ev
	.quad	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm6IprobeEii
	.quad	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm5ProbeEii
	.quad	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.quad	_ZNK3MPI4Comm9Get_groupEv
	.quad	_ZNK3MPI4Comm8Get_sizeEv
	.quad	_ZNK3MPI4Comm8Get_rankEv
	.quad	_ZNK3MPI9Intracomm5CloneEv
	.quad	_ZN3MPI4Comm4FreeEv
	.quad	_ZNK3MPI4Comm8Is_interEv
	.quad	_ZNK3MPI4Comm7BarrierEv
	.quad	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.quad	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.quad	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.quad	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.quad	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.quad	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.quad	_ZN3MPI4Comm10DisconnectEv
	.quad	_ZNK3MPI4Comm8Get_nameEPcRi
	.quad	_ZN3MPI4Comm8Set_nameEPKc
	.quad	_ZNK3MPI4Comm12Get_topologyEv
	.quad	_ZN3MPI4Comm5AbortEi
	.quad	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.quad	_ZNK3MPI4Comm14Get_errhandlerEv
	.quad	_ZNK3MPI4Comm8Set_attrEiPKv
	.quad	_ZNK3MPI4Comm8Get_attrEiPv
	.quad	_ZN3MPI4Comm11Delete_attrEi
	.quad	_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI9Intracomm6CreateERKNS_5GroupE
	.quad	_ZNK3MPI9Intracomm5SplitEii
	.quad	_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii
	.quad	_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb
	.quad	_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b
	.quad	_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi
	.quad	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi
	.quad	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi
	.size	_ZTVN3MPI9IntracommE, 568

	.type	_ZTSN3MPI9IntracommE,@object    # @_ZTSN3MPI9IntracommE
	.section	.rodata._ZTSN3MPI9IntracommE,"aG",@progbits,_ZTSN3MPI9IntracommE,comdat
	.weak	_ZTSN3MPI9IntracommE
_ZTSN3MPI9IntracommE:
	.asciz	"N3MPI9IntracommE"
	.size	_ZTSN3MPI9IntracommE, 17

	.type	_ZTIN3MPI9IntracommE,@object    # @_ZTIN3MPI9IntracommE
	.section	.data.rel.ro._ZTIN3MPI9IntracommE,"aGw",@progbits,_ZTIN3MPI9IntracommE,comdat
	.weak	_ZTIN3MPI9IntracommE
	.p2align	3
_ZTIN3MPI9IntracommE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI9IntracommE
	.quad	_ZTIN3MPI4CommE
	.size	_ZTIN3MPI9IntracommE, 24

	.type	_ZTVN3MPI8CartcommE,@object     # @_ZTVN3MPI8CartcommE
	.section	.data.rel.ro._ZTVN3MPI8CartcommE,"aGw",@progbits,_ZTVN3MPI8CartcommE,comdat
	.weak	_ZTVN3MPI8CartcommE
	.p2align	3
_ZTVN3MPI8CartcommE:
	.quad	0
	.quad	_ZTIN3MPI8CartcommE
	.quad	_ZN3MPI8CartcommD2Ev
	.quad	_ZN3MPI8CartcommD0Ev
	.quad	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm6IprobeEii
	.quad	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm5ProbeEii
	.quad	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.quad	_ZNK3MPI4Comm9Get_groupEv
	.quad	_ZNK3MPI4Comm8Get_sizeEv
	.quad	_ZNK3MPI4Comm8Get_rankEv
	.quad	_ZNK3MPI8Cartcomm5CloneEv
	.quad	_ZN3MPI4Comm4FreeEv
	.quad	_ZNK3MPI4Comm8Is_interEv
	.quad	_ZNK3MPI4Comm7BarrierEv
	.quad	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.quad	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.quad	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.quad	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.quad	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.quad	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.quad	_ZN3MPI4Comm10DisconnectEv
	.quad	_ZNK3MPI4Comm8Get_nameEPcRi
	.quad	_ZN3MPI4Comm8Set_nameEPKc
	.quad	_ZNK3MPI4Comm12Get_topologyEv
	.quad	_ZN3MPI4Comm5AbortEi
	.quad	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.quad	_ZNK3MPI4Comm14Get_errhandlerEv
	.quad	_ZNK3MPI4Comm8Set_attrEiPKv
	.quad	_ZNK3MPI4Comm8Get_attrEiPv
	.quad	_ZN3MPI4Comm11Delete_attrEi
	.quad	_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI9Intracomm6CreateERKNS_5GroupE
	.quad	_ZNK3MPI9Intracomm5SplitEii
	.quad	_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii
	.quad	_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb
	.quad	_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b
	.quad	_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi
	.quad	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi
	.quad	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi
	.quad	_ZNK3MPI8Cartcomm7Get_dimEv
	.quad	_ZNK3MPI8Cartcomm8Get_topoEiPiPbS1_
	.quad	_ZNK3MPI8Cartcomm13Get_cart_rankEPKi
	.quad	_ZNK3MPI8Cartcomm10Get_coordsEiiPi
	.quad	_ZNK3MPI8Cartcomm5ShiftEiiRiS1_
	.quad	_ZNK3MPI8Cartcomm3SubEPKb
	.quad	_ZNK3MPI8Cartcomm3MapEiPKiPKb
	.size	_ZTVN3MPI8CartcommE, 624

	.type	_ZTSN3MPI8CartcommE,@object     # @_ZTSN3MPI8CartcommE
	.section	.rodata._ZTSN3MPI8CartcommE,"aG",@progbits,_ZTSN3MPI8CartcommE,comdat
	.weak	_ZTSN3MPI8CartcommE
_ZTSN3MPI8CartcommE:
	.asciz	"N3MPI8CartcommE"
	.size	_ZTSN3MPI8CartcommE, 16

	.type	_ZTIN3MPI8CartcommE,@object     # @_ZTIN3MPI8CartcommE
	.section	.data.rel.ro._ZTIN3MPI8CartcommE,"aGw",@progbits,_ZTIN3MPI8CartcommE,comdat
	.weak	_ZTIN3MPI8CartcommE
	.p2align	3
_ZTIN3MPI8CartcommE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI8CartcommE
	.quad	_ZTIN3MPI9IntracommE
	.size	_ZTIN3MPI8CartcommE, 24

	.type	_ZTVN3MPI9GraphcommE,@object    # @_ZTVN3MPI9GraphcommE
	.section	.data.rel.ro._ZTVN3MPI9GraphcommE,"aGw",@progbits,_ZTVN3MPI9GraphcommE,comdat
	.weak	_ZTVN3MPI9GraphcommE
	.p2align	3
_ZTVN3MPI9GraphcommE:
	.quad	0
	.quad	_ZTIN3MPI9GraphcommE
	.quad	_ZN3MPI9GraphcommD2Ev
	.quad	_ZN3MPI9GraphcommD0Ev
	.quad	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm6IprobeEii
	.quad	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm5ProbeEii
	.quad	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.quad	_ZNK3MPI4Comm9Get_groupEv
	.quad	_ZNK3MPI4Comm8Get_sizeEv
	.quad	_ZNK3MPI4Comm8Get_rankEv
	.quad	_ZNK3MPI9Graphcomm5CloneEv
	.quad	_ZN3MPI4Comm4FreeEv
	.quad	_ZNK3MPI4Comm8Is_interEv
	.quad	_ZNK3MPI4Comm7BarrierEv
	.quad	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.quad	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.quad	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.quad	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.quad	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.quad	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.quad	_ZN3MPI4Comm10DisconnectEv
	.quad	_ZNK3MPI4Comm8Get_nameEPcRi
	.quad	_ZN3MPI4Comm8Set_nameEPKc
	.quad	_ZNK3MPI4Comm12Get_topologyEv
	.quad	_ZN3MPI4Comm5AbortEi
	.quad	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.quad	_ZNK3MPI4Comm14Get_errhandlerEv
	.quad	_ZNK3MPI4Comm8Set_attrEiPKv
	.quad	_ZNK3MPI4Comm8Get_attrEiPv
	.quad	_ZN3MPI4Comm11Delete_attrEi
	.quad	_ZNK3MPI9Intracomm4ScanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI9Intracomm6ExscanEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI9Intracomm6CreateERKNS_5GroupE
	.quad	_ZNK3MPI9Intracomm5SplitEii
	.quad	_ZNK3MPI9Intracomm16Create_intercommEiRKNS_4CommEii
	.quad	_ZNK3MPI9Intracomm11Create_cartEiPKiPKbb
	.quad	_ZNK3MPI9Intracomm12Create_graphEiPKiS2_b
	.quad	_ZNK3MPI9Intracomm6AcceptEPKcRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm7ConnectEPKcRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEi
	.quad	_ZNK3MPI9Intracomm5SpawnEPKcPS2_iRKNS_4InfoEiPi
	.quad	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEi
	.quad	_ZN3MPI9Intracomm14Spawn_multipleEiPPKcPS3_PKiPKNS_4InfoEiPi
	.quad	_ZNK3MPI9Graphcomm8Get_dimsEPiS1_
	.quad	_ZNK3MPI9Graphcomm8Get_topoEiiPiS1_
	.quad	_ZNK3MPI9Graphcomm19Get_neighbors_countEi
	.quad	_ZNK3MPI9Graphcomm13Get_neighborsEiiPi
	.quad	_ZNK3MPI9Graphcomm3MapEiPKiS2_
	.size	_ZTVN3MPI9GraphcommE, 608

	.type	_ZTSN3MPI9GraphcommE,@object    # @_ZTSN3MPI9GraphcommE
	.section	.rodata._ZTSN3MPI9GraphcommE,"aG",@progbits,_ZTSN3MPI9GraphcommE,comdat
	.weak	_ZTSN3MPI9GraphcommE
_ZTSN3MPI9GraphcommE:
	.asciz	"N3MPI9GraphcommE"
	.size	_ZTSN3MPI9GraphcommE, 17

	.type	_ZTIN3MPI9GraphcommE,@object    # @_ZTIN3MPI9GraphcommE
	.section	.data.rel.ro._ZTIN3MPI9GraphcommE,"aGw",@progbits,_ZTIN3MPI9GraphcommE,comdat
	.weak	_ZTIN3MPI9GraphcommE
	.p2align	3
_ZTIN3MPI9GraphcommE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI9GraphcommE
	.quad	_ZTIN3MPI9IntracommE
	.size	_ZTIN3MPI9GraphcommE, 24

	.type	_ZTVN3MPI9IntercommE,@object    # @_ZTVN3MPI9IntercommE
	.section	.data.rel.ro._ZTVN3MPI9IntercommE,"aGw",@progbits,_ZTVN3MPI9IntercommE,comdat
	.weak	_ZTVN3MPI9IntercommE
	.p2align	3
_ZTVN3MPI9IntercommE:
	.quad	0
	.quad	_ZTIN3MPI9IntercommE
	.quad	_ZN3MPI9IntercommD2Ev
	.quad	_ZN3MPI9IntercommD0Ev
	.quad	_ZNK3MPI4Comm4SendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm4RecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5BsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5SsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5RsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IbsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IssendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IrsendEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm5IrecvEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm6IprobeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm6IprobeEii
	.quad	_ZNK3MPI4Comm5ProbeEiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm5ProbeEii
	.quad	_ZNK3MPI4Comm9Send_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Bsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Ssend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm10Rsend_initEPKviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm9Recv_initEPviRKNS_8DatatypeEii
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_iiRNS_6StatusE
	.quad	_ZNK3MPI4Comm8SendrecvEPKviRKNS_8DatatypeEiiPviS5_ii
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiiiRNS_6StatusE
	.quad	_ZNK3MPI4Comm16Sendrecv_replaceEPviRKNS_8DatatypeEiiii
	.quad	_ZNK3MPI4Comm9Get_groupEv
	.quad	_ZNK3MPI4Comm8Get_sizeEv
	.quad	_ZNK3MPI4Comm8Get_rankEv
	.quad	_ZNK3MPI9Intercomm5CloneEv
	.quad	_ZN3MPI4Comm4FreeEv
	.quad	_ZNK3MPI4Comm8Is_interEv
	.quad	_ZNK3MPI4Comm7BarrierEv
	.quad	_ZNK3MPI4Comm5BcastEPviRKNS_8DatatypeEi
	.quad	_ZNK3MPI4Comm6GatherEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm7GathervEPKviRKNS_8DatatypeEPvPKiS8_S5_i
	.quad	_ZNK3MPI4Comm7ScatterEPKviRKNS_8DatatypeEPviS5_i
	.quad	_ZNK3MPI4Comm8ScattervEPKvPKiS4_RKNS_8DatatypeEPviS7_i
	.quad	_ZNK3MPI4Comm9AllgatherEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm10AllgathervEPKviRKNS_8DatatypeEPvPKiS8_S5_
	.quad	_ZNK3MPI4Comm8AlltoallEPKviRKNS_8DatatypeEPviS5_
	.quad	_ZNK3MPI4Comm9AlltoallvEPKvPKiS4_RKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm9AlltoallwEPKvPKiS4_PKNS_8DatatypeEPvS4_S4_S7_
	.quad	_ZNK3MPI4Comm6ReduceEPKvPviRKNS_8DatatypeERKNS_2OpEi
	.quad	_ZNK3MPI4Comm9AllreduceEPKvPviRKNS_8DatatypeERKNS_2OpE
	.quad	_ZNK3MPI4Comm14Reduce_scatterEPKvPvPiRKNS_8DatatypeERKNS_2OpE
	.quad	_ZN3MPI4Comm10DisconnectEv
	.quad	_ZNK3MPI4Comm8Get_nameEPcRi
	.quad	_ZN3MPI4Comm8Set_nameEPKc
	.quad	_ZNK3MPI4Comm12Get_topologyEv
	.quad	_ZN3MPI4Comm5AbortEi
	.quad	_ZN3MPI4Comm14Set_errhandlerERKNS_10ErrhandlerE
	.quad	_ZNK3MPI4Comm14Get_errhandlerEv
	.quad	_ZNK3MPI4Comm8Set_attrEiPKv
	.quad	_ZNK3MPI4Comm8Get_attrEiPv
	.quad	_ZN3MPI4Comm11Delete_attrEi
	.quad	_ZNK3MPI9Intercomm15Get_remote_sizeEv
	.quad	_ZNK3MPI9Intercomm16Get_remote_groupEv
	.quad	_ZNK3MPI9Intercomm5MergeEb
	.quad	_ZNK3MPI9Intercomm6CreateERKNS_5GroupE
	.quad	_ZNK3MPI9Intercomm5SplitEii
	.size	_ZTVN3MPI9IntercommE, 504

	.type	_ZTSN3MPI9IntercommE,@object    # @_ZTSN3MPI9IntercommE
	.section	.rodata._ZTSN3MPI9IntercommE,"aG",@progbits,_ZTSN3MPI9IntercommE,comdat
	.weak	_ZTSN3MPI9IntercommE
_ZTSN3MPI9IntercommE:
	.asciz	"N3MPI9IntercommE"
	.size	_ZTSN3MPI9IntercommE, 17

	.type	_ZTIN3MPI9IntercommE,@object    # @_ZTIN3MPI9IntercommE
	.section	.data.rel.ro._ZTIN3MPI9IntercommE,"aGw",@progbits,_ZTIN3MPI9IntercommE,comdat
	.weak	_ZTIN3MPI9IntercommE
	.p2align	3
_ZTIN3MPI9IntercommE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSN3MPI9IntercommE
	.quad	_ZTIN3MPI4CommE
	.size	_ZTIN3MPI9IntercommE, 24

	.type	_ZTVN3MPI4InfoE,@object         # @_ZTVN3MPI4InfoE
	.section	.data.rel.ro._ZTVN3MPI4InfoE,"aGw",@progbits,_ZTVN3MPI4InfoE,comdat
	.weak	_ZTVN3MPI4InfoE
	.p2align	3
_ZTVN3MPI4InfoE:
	.quad	0
	.quad	_ZTIN3MPI4InfoE
	.quad	_ZN3MPI4InfoD2Ev
	.quad	_ZN3MPI4InfoD0Ev
	.quad	_ZN3MPI4Info6DeleteEPKc
	.quad	_ZNK3MPI4Info3DupEv
	.quad	_ZN3MPI4Info4FreeEv
	.quad	_ZNK3MPI4Info3GetEPKciPc
	.quad	_ZNK3MPI4Info9Get_nkeysEv
	.quad	_ZNK3MPI4Info10Get_nthkeyEiPc
	.quad	_ZNK3MPI4Info12Get_valuelenEPKcRi
	.quad	_ZN3MPI4Info3SetEPKcS2_
	.size	_ZTVN3MPI4InfoE, 96

	.type	_ZTSN3MPI4InfoE,@object         # @_ZTSN3MPI4InfoE
	.section	.rodata._ZTSN3MPI4InfoE,"aG",@progbits,_ZTSN3MPI4InfoE,comdat
	.weak	_ZTSN3MPI4InfoE
_ZTSN3MPI4InfoE:
	.asciz	"N3MPI4InfoE"
	.size	_ZTSN3MPI4InfoE, 12

	.type	_ZTIN3MPI4InfoE,@object         # @_ZTIN3MPI4InfoE
	.section	.data.rel.ro._ZTIN3MPI4InfoE,"aGw",@progbits,_ZTIN3MPI4InfoE,comdat
	.weak	_ZTIN3MPI4InfoE
	.p2align	3
_ZTIN3MPI4InfoE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI4InfoE
	.size	_ZTIN3MPI4InfoE, 16

	.type	_ZTVN3MPI2OpE,@object           # @_ZTVN3MPI2OpE
	.section	.data.rel.ro._ZTVN3MPI2OpE,"aGw",@progbits,_ZTVN3MPI2OpE,comdat
	.weak	_ZTVN3MPI2OpE
	.p2align	3
_ZTVN3MPI2OpE:
	.quad	0
	.quad	_ZTIN3MPI2OpE
	.quad	_ZN3MPI2OpD2Ev
	.quad	_ZN3MPI2OpD0Ev
	.quad	_ZN3MPI2Op4InitEPFvPKvPviRKNS_8DatatypeEEb
	.quad	_ZN3MPI2Op4FreeEv
	.quad	_ZNK3MPI2Op12Reduce_localEPKvPviRKNS_8DatatypeE
	.quad	_ZNK3MPI2Op14Is_commutativeEv
	.size	_ZTVN3MPI2OpE, 64

	.type	_ZTSN3MPI2OpE,@object           # @_ZTSN3MPI2OpE
	.section	.rodata._ZTSN3MPI2OpE,"aG",@progbits,_ZTSN3MPI2OpE,comdat
	.weak	_ZTSN3MPI2OpE
_ZTSN3MPI2OpE:
	.asciz	"N3MPI2OpE"
	.size	_ZTSN3MPI2OpE, 10

	.type	_ZTIN3MPI2OpE,@object           # @_ZTIN3MPI2OpE
	.section	.data.rel.ro._ZTIN3MPI2OpE,"aGw",@progbits,_ZTIN3MPI2OpE,comdat
	.weak	_ZTIN3MPI2OpE
	.p2align	3
_ZTIN3MPI2OpE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTSN3MPI2OpE
	.size	_ZTIN3MPI2OpE, 16

	.type	_ZTVN3MPI9Comm_NullE,@object    # @_ZTVN3MPI9Comm_NullE
	.section	.data.rel.ro._ZTVN3MPI9Comm_NullE,"aGw",@progbits,_ZTVN3MPI9Comm_NullE,comdat
	.weak	_ZTVN3MPI9Comm_NullE
	.p2align	3
_ZTVN3MPI9Comm_NullE:
	.quad	0
	.quad	_ZTIN3MPI9Comm_NullE
	.quad	_ZN3MPI9Comm_NullD2Ev
	.quad	_ZN3MPI9Comm_NullD0Ev
	.size	_ZTVN3MPI9Comm_NullE, 32

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z6vecAddPfS_S_i"
	.size	.L__unnamed_1, 17

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z12forward_passPfS_S_ii"
	.size	.L__unnamed_2, 25

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z13backward_passPfS_S_i"
	.size	.L__unnamed_3, 25

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z10calc_gradsPfS_S_ii"
	.size	.L__unnamed_4, 23

	.type	.L__unnamed_5,@object           # @4
	.section	.nv_fatbin,"a",@progbits
	.p2align	3
.L__unnamed_5:
	.asciz	"P\355U\272\001\000\020\000\240\\\000\000\000\000\000\000\002\000\001\001@\000\000\000\230R\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\001\0002\000\000\000\000\000\000\000\000\000\000\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\0013\007\000\000\000\000\000\000\000\002\000\276\000r\000\000\000\000\000\000\000\000\000\000\000\360Q\000\000\000\000\000\000\360L\000\000\000\000\000\0002\0052\000@\0008\000\003\000@\000\024\000\001\000\000.shstrtab\000.strtab\000.symtab\000.symtab_shndx\000.nv.info\000.text._Z10calc_gradsPfS_S_ii\000.nv.info._Z10calc_gradsPfS_S_ii\000.nv.shared._Z10calc_gradsPfS_S_ii\000.nv.global\000.nv.global.init\000.nv.constant0._Z10calc_gradsPfS_S_ii\000.text._Z13backward_passPfS_S_i\000.nv.info._Z13backward_passPfS_S_i\000.nv.shared._Z13backward_passPfS_S_i\000.nv.constant0._Z13backward_passPfS_S_i\000.text._Z12forward_passPfS_S_ii\000.nv.info._Z12forward_passPfS_S_ii\000.nv.shared._Z12forward_passPfS_S_ii\000.nv.constant0._Z12forward_passPfS_S_ii\000.text._Z6vecAddPfS_S_i\000.nv.info._Z6vecAddPfS_S_i\000.nv.shared._Z6vecAddPfS_S_i\000.nv.constant0._Z6vecAddPfS_S_i\000.nv.rel.action\000\000.shstrtab\000.strtab\000.symtab\000.symtab_shndx\000.nv.info\000_Z10calc_gradsPfS_S_ii\000.text._Z10calc_gradsPfS_S_ii\000.nv.info._Z10calc_gradsPfS_S_ii\000.nv.shared._Z10calc_gradsPfS_S_ii\000.nv.global\000blockIdx\000blockDim\000threadIdx\000.nv.global.init\000_$_str\000.nv.constant0._Z10calc_gradsPfS_S_ii\000_param\000_Z13backward_passPfS_S_i\000.text._Z13backward_passPfS_S_i\000.nv.info._Z13backward_passPfS_S_i\000.nv.shared._Z13backward_passPfS_S_i\000.nv.constant0._Z13backward_passPfS_S_i\000_Z12forward_passPfS_S_ii\000.text._Z12forward_passPfS_S_ii\000.nv.info._Z12forward_passPfS_S_ii\000.nv.shared._Z12forward_passPfS_S_ii\000$_Z12forward_passPfS_S_ii$__cuda_sm20_rcp_rn_f32_slowpath\000.nv.constant0._Z12forward_passPfS_S_ii\000_Z6vecAddPfS_S_i\000.text._Z6vecAddPfS_S_i\000.nv.info._Z6vecAddPfS_S_i\000.nv.shared._Z6vecAddPfS_S_i\000.nv.constant0._Z6vecAddPfS_S_i\000.nv.rel.action\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000I\000\000\000\003\000\016\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\250\000\000\000\003\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\263\000\000\000\001\000\023\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\274\000\000\000\001\000\023\000\002\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\305\000\000\000\001\000\023\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\317\000\000\000\003\000\022\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\337\000\000\000\001\000\022\000\000\000\000\000\000\000\000\000\013\000\000\000\000\000\000\000\346\000\000\000\003\000\n\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000+\001\000\000\003\000\017\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\220\001\000\000\003\000\013\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320\001\000\000\003\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\002\000\000\"\000\020\000\020\022\000\000\000\000\000\0000\004\000\000\000\000\000\000o\002\000\000\003\000\f\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\247\002\000\000\003\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\364\002\000\000\003\000\r\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\023\003\000\000\003\000\t\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0002\000\000\000\022\020\016\000\000\000\000\000\000\000\000\000@\017\000\000\000\000\000\000\022\001\000\000\022\020\017\000\000\000\000\000\000\000\000\000\300\f\000\000\000\000\000\000\267\001\000\000\022\020\020\000\000\000\000\000\000\000\000\000@\026\000\000\000\000\000\000\226\002\000\000\022\020\021\000\000\000\000\000\000\000\000\000\000\t\000\000\000\000\000\000\004/\b\000\024\000\000\000\r\000\000\000\004#\b\000\024\000\000\000\000\000\000\000\004\022\b\000\024\000\000\000 \000\000\000\004\021\b\000\024\000\000\000 \000\000\000\004/\b\000\023\000\000\000\016\000\000\000\004#\b\000\f\000\000\000\000\000\000\000\004\022\b\000\f\000\000\000\000\000\000\000\004\021\b\000\f\000\000\000\000\000\000\000\004#\b\000\023\000\000\000\000\000\000\000\004\022\b\000\023\000\000\0008\000\000\000\004\021\b\000\023\000\000\0008\000\000\000\004/\b\000\022\000\000\000\r\000\000\000\004#\b\000\022\000\000\000\000\000\000\000\004\022\b\000\022\000\000\000(\000\000\000\004\021\b\000\022\000\000\000(\000\000\000\004/\b\000\021\000\000\000\016\000\000\000\004#\b\000\021\000\000\000\000\000\000\000\004\022\b\000\021\000\000\000(\000\000\000\004\021\b\000\021\000\000\000(\000\000\000\0047\004\000r\000\000\000\0010\000\000\001*\000\000\004\n\b\000\b\000\000\000@\001 \000\003\031 \000\004\027\f\000\000\000\000\000\004\000\034\000\000\360\021\000\004\027\f\000\000\000\000\000\003\000\030\000\000\360\021\000\004\027\f\000\000\000\000\000\002\000\020\000\000\360!\000\004\027\f\000\000\000\000\000\001\000\b\000\000\360!\000\004\027\f\000\000\000\000\000\000\000\000\000\000\360!\000\003\033\377\000\004\035\004\000h\004\000\000\004\034\004\000\020\017\000\000\0044\020\000\310\b\000\000\000\000\000\000\001\000\000\000\000\017\000\000\004\036\004\000 \002\000\000\0047\004\000r\000\000\000\0010\000\000\001*\000\000\004\n\b\000\n\000\000\000@\001\034\000\003\031\034\000\004\027\f\000\000\000\000\000\003\000\030\000\000\360\021\000\004\027\f\000\000\000\000\000\002\000\020\000\000\360!\000\004\027\f\000\000\000\000\000\001\000\b\000\000\360!\000\004\027\f\000\000\000\000\000\000\000\000\000\000\360!\000\003\033\377\000\004\035\004\000\350\003\000\000\004\034\004\000\210\f\000\000\004\036\004\000 \000\000\000\0047\004\000r\000\000\000\0010\000\000\001*\000\000\004\n\b\000\r\000\000\000@\001 \000\003\031 \000\004\027\f\000\000\000\000\000\004\000\034\000\000\360\021\000\004\027\f\000\000\000\000\000\003\000\030\000\000\360\021\000\004\027\f\000\000\000\000\000\002\000\020\000\000\360!\000\004\027\f\000\000\000\000\000\001\000\b\000\000\360!\000\004\027\f\000\000\000\000\000\000\000\000\000\000\360!\000\003\033\377\000\004\035\004\000h\004\000\000\004\034\004\000\b\022\000\000\0044p\000\260\007\000\000\000\000\000\000\001\000\000\000 \r\000\000\b\020\000\000\000\000\000\000\001\000\000\000X\020\000\000P\020\000\000\000\000\000\000\001\000\000\000X\020\000\000\350\022\000\000\000\000\000\000\001\000\000\000\000\026\000\000x\023\000\000\000\000\000\000\001\000\000\000\000\026\000\000\330\025\000\000\000\000\000\000\001\000\000\000\000\026\000\000\370\025\000\000\000\000\000\000\001\000\000\000\000\026\000\000\004\036\004\000 \002\000\000\0047\004\000r\000\000\000\0010\000\000\001*\000\000\004\n\b\000\017\000\000\000@\001\034\000\003\031\034\000\004\027\f\000\000\000\000\000\003\000\030\000\000\360\021\000\004\027\f\000\000\000\000\000\002\000\020\000\000\360!\000\004\027\f\000\000\000\000\000\001\000\b\000\000\360!\000\004\027\f\000\000\000\000\000\000\000\000\000\000\360!\000\003\033\377\000\004\035\004\000\350\003\000\000\004\034\004\000\270\b\000\000\004\036\004\000 \000\000\000\000\000\000\000K\000\000\000\000\000\000\000\000\002\002\b\020\n/\"\000\000\000\b\000\000\000\000\000\000\b\b\000\000\000\000\000\000\020\b\000\000\000\000\000\000\030\b\000\000\000\000\000\000 \b\000\000\000\000\000\000(\b\000\000\000\000\000\0000\b\000\000\000\000\000\0008\b\000\000\000\000\001\000\000\b\000\000\000\000\001\000\b\b\000\000\000\000\001\000\020\b\000\000\000\000\001\000\030\b\000\000\000\000\001\000 \b\000\000\000\000\001\000(\b\000\000\000\000\001\0000\b\000\000\000\000\001\0008\b\000\000\000\000\002\000\000\b\000\000\000\000\002\000\b\b\000\000\000\000\002\000\020\b\000\000\000\000\002\000\030\b\000\000\000\000\002\000 \b\000\000\000\000\002\000(\b\000\000\000\000\002\0000\b\000\000\000\000\002\0008\b\000\000\000\000\000\000\000\024,\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\357\037\340\375\003<d\000\001\000\207\000\200\007\230L\001\001\207\375\377\377\017\034\000\000w\003\000\000\310\360\357\037\340\375\003\274\177\000\007\001\007\000\200\003l[\017\000\200\000\000\000@\342\300\000\020\000\000\000\240\343\357\037\340!\003\274\177\000\000\001\367\017\000\000\020\\\000\n\007\000\000\000\340\\\002\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\367\017\200\007\230\\\000\000'\000\200\007\230\\\004\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\007\000\200\007\230\\\004\000G\000\200\007\230\\\002\000\027\000\200\007\230L\357\037\340\375\003\274\177\000\000\000\367\017\200\007\230\\\002\003'\000\000\002G\\\000\004\007\000\000\002G\\\357\037\340!\003\274\177\000\003\360\307\025\000\000\000\001\003\003\007\000\000\000\224\357\003\0007\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\207\025\000\000\000\001\004\004\007\000\000\000\224\357\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\006\360\007\025\000\000\000\001\006\006\007\000\000\000\225\357\005\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\f\000w\000\200\007\230\\\005\000W\000\200\007\230\\\f\000\307\000\200\007\230\\\357\037\340!\003\274\177\000\006\360\207\024\000\000\000\001\006\006\007\000\000\000\225\357\b\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340!\003\274\177\000\006\360\007\024\000\000\000\001\006\006\007\000\000\000\225\357\n\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\013\000w\000\200\007\230\\\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000W\000\200\007\230\\\007\000\307\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\n\000\247\000\200\007\230\\\357\037\340\375\003\274\177\000\013\000\267\000\200\007\230\\\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\r\002\367\017\000\200\020\\\005\000\367\017\000\b\020\\\r\000\327\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\f\r\367\017\000\200\327[\r\r\367\017\300\002\330[\357\037\340\375\003\274g\000\f\000\307\000\200\007\230\\\r\000\327\000\200\007\230\\\n\f\007\000\000\000\260\240\357\037\340\375\003\274\177\000\013\002\207\000\000\000\020\034\005\000\367\017\000\b\020\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\n\013\367\017\000\200\327[\013\013\367\017\300\002\330[\357\037\340\375\003\274g\000\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\b\n\007\000\000\000\260\240\357\037\340\375\003\274\177\000\t\002\007\001\000\000\020\034\005\000\367\017\000\b\020\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\b\t\367\017\000\200\327[\t\t\367\017\300\002\330[\357\037\340\375\003\274g\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\006\b\007\000\000\000\260\240\357\037\340\375\003\274\177\000\007\002\207\001\000\000\020\034\005\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\007\367\017\000\200\327[\007\007\367\017\300\002\330[\357\037\340\375\003\274g\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\004\006\007\000\000\000\220\240\357\037\340\375\003\274\177\000\005\002\307\001\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\031\340\375\003\274\177\000\003\000W\002\000\000\310\360\003\0007\000\200\007\230\\\004\000'\000\200\007\230L\017\031\340!\003\274\177\000\003\003G\000\000\0038\\\004\000\027\002\000\000\310\360\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\003\003G\000\000\000\020\\\005\002\007\002\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\031\340\375\003\274\177\000\003\004\007\000\000\000\220\240\005\002\007\002\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\003\004\007\000\000\000\220\200\005\002\307\001\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\007\003G\000\200\003m[\002\000'\000\200\007\230\\\357\037\340\375\003\274\177\000\000\000\007\000\200\007\230\\\017\000\200\221\000\000@\342\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\005\002\007\001\000\000\020\034\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\006\000G\000\200\007\230\\\007\000W\000\200\007\230\\\005\002\007\002\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003<d\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\004:G\000\000\000\340\\\357\037\340\375\003\274\177\000\003\004\367\001\000\000)8\004\000G\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\004'\000\300\001\3706\005\004'\000\000\000H8\005\006W\000\000\200\020\\\357\037\340\375\003\274\177\000\003\0077\000\000\b\020\\\006\000\367\017\200\007\230\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\006\004\007\000\000\000\220\240\357\037\340\375\003\274\177\000\005\002G\002\000\000\020\034\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\006\004\007\000\000\000\220\240\357\037\340\375\003\274\177\000\000\000\000q\000\000\220\342\017\000\007\000\000\000@\342\005\002G\002\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\200\005\002\207\001\000\000\020\034\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\007\003G\000\200\003m[\357\037\340\375\003\274\177\000\017\000\000\000\000\000\370\360\017\000\007\000\000\000@\342\005\002\207\000\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\b\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000W\000\200\007\230\\\005\002G\002\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340!\003\274\177\000\005\004\007\000\000\000\220\200\006:W\000\000\000\340\\\007\006\367\001\000\000)8\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\003\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000w\000\200\007\230\\\007\0007\000\200\007\230\\\003\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\003\007'\000\300\001\3706\007\007'\000\000\000H8\007\bw\000\000\200\020\\\357\037\340\375\003\274\177\000\003\t7\000\000\b\020\\\007\000w\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017\300\001\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\003\006\007\000\000\000\220\200\007\002\367\017\000\200\020\\\357\037\340\375\003\274\177\000\004\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017@\002\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\006\006\007\000\000\000\260\200\004\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000w\000\200\007\230\\\t\002\307\001\000\000\020\034\007\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\t\000\227\000\200\007\230\\\007\000w\000\200\007\230\\\b\t\367\017\000\200\327[\357\037\340\375\003\274\177\000\t\t\367\017\300\003\330[\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\017\031\340!\003\274\177\000\b\b\007\000\000\000\220\200\007\005\207\000\000\0038\\\t\002\007\002\000\000\020\034\357\037\340\375\003\274\177\000\005\000\367\017\000\b\020\\\t\000\227\000\200\007\230\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\b\t\367\017\000\200\327[\t\t\367\017\300\002\330[\b\000\207\000\200\007\230\\\357\037\340!\003\274\177\000\t\000\227\000\200\007\230\\\005\b\007\000\000\000\220\200\007\007W\000\000\000\020\\\017\031\340\375\003\274\177\000\007:w\000\000\000\340\\\b\007\367\001\000\000)8\t\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000w\000\200\007\230\\\007\000\207\000\200\007\230\\\b\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\b\007'\000@\004\3706\357\037\340\375\003\274\177\000\007\007'\000\000\000H8\004\004w\000\000\200\020\\\006\006\207\000\000\b\020\\\357\037\340\375\003\274\177\000\007\000G\000\200\007\230\\\004\000g\000\200\007\230\\\006\007\367\017\000\200\327[\357\037\340\375\003\274\177\000\007\007\367\017@\002\330[\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\017\031\340\375\003\274\177\000\004\006\007\000\000\000\220\200\007\002\007\001\000\000\020\034\b\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\006\007\367\017\000\200\327[\357\037\340\375\003\274\177\000\007\007\367\017@\004\330[\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\017\031\340\375\003\274\177\000\006\006\007\000\000\000\260\200\n\000g\000\200\007\230\\\013\000w\000\200\007\230\\\017\031\340\375\003\274\177\000\005:W\000\000\000\340\\\006\005\367\001\000\000)8\007\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000W\000\200\007\230\\\b\000g\000\200\007\230\\\t\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\t\b'\000\300\004\3706\357\037\340\375\003\274\177\000\b\b'\000\000\000H8\b\n\207\000\000\200\020\\\t\013\227\000\000\b\020\\\357\037\340\375\003\274\177\000\007\000\207\000\200\007\230\\\005\000\227\000\200\007\230\\\006\007\367\017\000\200\327[\357\037\340\375\003\274\177\000\007\007\367\017\300\002\330[\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\017\031\340\375\003\274\177\000\006\006\007\000\000\000\220\200\003\003G\000\000\003\200Y\005\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000\227\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\357\037\340\375\003\274\177\000\017\000\007\000\000\000@\342\005\002G\002\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\003\004\027\000\000\000\000\034\005\002G\002\000\000\020\034\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340=\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\000\207\217\377\017@\342\357\037\340\375\003\374\037\000\017\000\007\000\000\000@\342\017\000\007\000\000\000\000\343\017\000\207\377\377\017@\342\340\007\000\374\000\200\037\000\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P\357\037\340\375\003<d\000\001\000\207\000\200\007\230L\001\001\207\375\377\377\017\034\000\000w\003\000\000\310\360\357\037\340\375\003\274\177\000\007\001\007\000\200\003l[\017\000\200\000\000\000@\342\300\000\020\000\000\000\240\343\357\037\340!\003\274\177\000\000\001\367\017\000\000\020\\\000\n\007\000\000\000\340\\\002\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\367\017\200\007\230\\\000\000'\000\200\007\230\\\004\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\007\000\200\007\230\\\004\000G\000\200\007\230\\\002\000\027\000\200\007\230L\357\037\340\375\003\274\177\000\000\000\367\017\200\007\230\\\002\003'\000\000\002G\\\000\004\007\000\000\002G\\\357\037\340!\003\274\177\000\003\360\207\025\000\000\000\001\003\003\007\000\000\000\224\357\003\0007\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\007\025\000\000\000\001\004\004\007\000\000\000\225\357\n\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\013\000W\000\200\007\230\\\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\207\024\000\000\000\001\004\004\007\000\000\000\225\357\006\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000W\000\200\007\230\\\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\007\024\000\000\000\001\004\004\007\000\000\000\225\357\b\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000W\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\004\000\247\000\200\007\230\\\005\000\267\000\200\007\230\\\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\013\002\367\017\000\200\020\\\f\000\367\017\000\b\020\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\f\000\307\000\200\007\230\\\n\013\367\017\000\200\327[\013\013\367\017@\006\330[\357\037\340\375\003\274g\000\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\b\n\007\000\000\000\260\240\357\037\340\375\003\274\177\000\t\002\207\000\000\000\020\034\n\000\367\017\000\b\020\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\n\000\247\000\200\007\230\\\b\t\367\017\000\200\327[\t\t\367\017@\005\330[\357\037\340\375\003\274g\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\006\b\007\000\000\000\260\240\357\037\340\375\003\274\177\000\007\002\007\001\000\000\020\034\b\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\006\007\367\017\000\200\327[\007\007\367\017@\004\330[\357\037\340\375\003\274g\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\004\006\007\000\000\000\260\240\357\037\340\375\003\274\177\000\005\002\207\001\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\031\340\375\003\274\177\000\003\000W\002\000\000\310\360\003\0007\000\200\007\230\\\004\000'\000\200\007\230L\017\031\340!\003\274\177\000\003\003G\000\000\0038\\\004\000\027\002\000\000\310\360\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\003\003G\000\000\000\020\\\005\002\307\001\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\031\340\375\003\274\177\000\003\004\007\000\000\000\220\240\005\002\307\001\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\003\004\007\000\000\000\220\200\005\002\207\001\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\007\003G\000\200\003m[\002\000'\000\200\007\230\\\357\037\340\375\003\274\177\000\000\000\007\000\200\007\230\\\017\000\200p\000\000@\342\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\005\002\367\017\000\200\020\\\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\006\000G\000\200\007\230\\\t\000W\000\200\007\230\\\005\002\307\001\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003<d\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\004:G\000\000\000\340\\\357\037\340\375\003\274\177\000\003\004\367\001\000\000)8\004\000G\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\b\004'\000\300\001\3706\007\004'\000\000\000H8\005\006w\000\000\200\020\\\357\037\340\375\003\274\177\000\003\t\207\000\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\200\005\002\207\000\000\000\020\034\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\006\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\004\000W\000\200\007\230\\\005\006w\000\000\200\020\\\006\004\207\000\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\003\003G\000\000 X\\\003\0037\000\000\000X\\\357\037\340\375\003\274\177\000\005\002\007\002\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\357\037\340\375\003\274\177\000\005\002\367\017\000\200\020\\\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\006\000G\000\200\007\230\\\007\000W\000\200\007\230\\\005\002\307\001\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003<d\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\004:G\000\000\000\340\\\357\037\340\375\003\274\177\000\003\004\367\001\000\000)8\004\000G\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\004'\000\300\001\3706\005\004'\000\000\000H8\005\006W\000\000\200\020\\\357\037\340\375\003\274\177\000\003\0077\000\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\003\360\007\000\000\370\003\001\357\037\340\375\003\274\177\000\003\003G\000\000 X\\\003\0047\000\000\000h\\\005\002G\002\000\000\020\034\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340=\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\005\002\007\002\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\200\005\002G\002\000\000\020\034\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\003\003G\000\000\000h\\\357\037\340\375\003\274\177\000\005\002\007\001\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\006\000G\000\200\007\230\\\007\000W\000\200\007\230\\\005\002\307\001\000\000\020\034\357\037\340\375\003\274\177\000\000\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\000\330[\004\000G\000\200\007\230\\\357\037\340!\003<d\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\004:G\000\000\000\340\\\357\037\340\375\003\274\177\000\000\004\367\001\000\000)8\004\000G\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\000\004'\000@\000\3706\005\004'\000\000\000H8\005\006W\000\000\200\020\\\357\037\340\375\003\274\177\000\000\007\007\000\000\b\020\\\005\000W\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\000\330[\004\000G\000\200\007\230\\\357\037\340=\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\000\007\000\000\000@\342\357\037\340\377\000\200\037\000\017\000\007\000\000\000\000\343\017\000\207\377\377\017@\342\000\017\007\000\000\000\260P\340\007\000\374\000\200\037\000\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P\357\037\340\375\003<d\000\001\000\207\000\200\007\230L\001\001\207\374\377\377\017\034\000\000w\003\000\000\310\360\357\037\340\375\003\274\177\000\007\001\007\000\200\003l[\017\000\200\000\000\000@\342\300\000\020\000\000\000\240\343\357\037\340!\003\274\177\000\000\001\367\017\000\000\020\\\000\n\007\000\000\000\340\\\002\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\367\017\200\007\230\\\000\000'\000\200\007\230\\\004\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\007\000\200\007\230\\\004\000G\000\200\007\230\\\002\000\027\000\200\007\230L\357\037\340\375\003\274\177\000\000\000\367\017\200\007\230\\\002\003'\000\000\002G\\\000\004\007\000\000\002G\\\357\037\340!\003\274\177\000\003\360\307\025\000\000\000\001\003\003\007\000\000\000\224\357\003\0007\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\207\025\000\000\000\001\004\004\007\000\000\000\224\357\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\006\360\007\025\000\000\000\001\006\006\007\000\000\000\225\357\005\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\f\000w\000\200\007\230\\\005\000W\000\200\007\230\\\f\000\307\000\200\007\230\\\357\037\340!\003\274\177\000\006\360\207\024\000\000\000\001\006\006\007\000\000\000\225\357\b\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340!\003\274\177\000\006\360\007\024\000\000\000\001\006\006\007\000\000\000\225\357\n\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\013\000w\000\200\007\230\\\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000W\000\200\007\230\\\007\000\307\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\n\000\247\000\200\007\230\\\357\037\340\375\003\274\177\000\013\000\267\000\200\007\230\\\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\r\002\207\000\000\000\020\034\005\000\367\017\000\b\020\\\r\000\327\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\f\r\367\017\000\200\327[\r\r\367\017\300\002\330[\357\037\340\375\003\274g\000\f\000\307\000\200\007\230\\\r\000\327\000\200\007\230\\\n\f\007\000\000\000\260\240\357\037\340\375\003\274\177\000\013\002\007\001\000\000\020\034\005\000\367\017\000\b\020\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\n\013\367\017\000\200\327[\013\013\367\017\300\002\330[\357\037\340\375\003\274g\000\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\b\n\007\000\000\000\260\240\357\037\340\375\003\274\177\000\t\002\207\001\000\000\020\034\005\000\367\017\000\b\020\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\b\t\367\017\000\200\327[\t\t\367\017\300\002\330[\357\037\340\375\003\274g\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\006\b\007\000\000\000\260\240\357\037\340\375\003\274\177\000\007\002\007\002\000\000\020\034\005\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\007\367\017\000\200\327[\007\007\367\017\300\002\330[\357\037\340\375\003\274g\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\004\006\007\000\000\000\220\240\357\037\340\375\003\274\177\000\005\002G\002\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\031\340\375\003\274\177\000\003\000W\002\000\000\310\360\003\0007\000\200\007\230\\\004\000'\000\200\007\230L\017\031\340!\003\274\177\000\003\003G\000\000\0038\\\004\000\027\002\000\000\310\360\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\003\003G\000\000\000\020\\\005\002\207\002\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\031\340\375\003\274\177\000\003\004\007\000\000\000\220\240\005\002\207\002\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\003\004\007\000\000\000\220\200\005\002\007\002\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\007\003G\000\200\003m[\002\000'\000\200\007\230\\\357\037\340\375\003\274\177\000\000\000\007\000\200\007\230\\\017\000\200\300\000\000@\342\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\006\000\367\017\200\007\230\\\005\002\307\002\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\031\340\375\003\274\177\000\006\004\007\000\000\000\220\240\005\002\007\003\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\031\340\375\003\274\177\000\006\004\007\000\000\000\220\240\000\000\200d\000\000\220\342\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\005\002\007\003\000\000\020\034\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\200\357\037\340\375\003\274\177\000\005\002G\002\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\357\037\340\375\003\274\177\000\007\003G\000\200\003m[\017\000\000\000\000\000\370\360\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\005\002\207\000\000\000\020\034\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\003\000G\000\200\007\230\\\005\000W\000\200\007\230\\\007\002\207\002\000\000\020\034\357\037\340\375\003\274\177\000\004\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017@\002\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\004\006\007\000\000\000\220\200\007\002G\002\000\000\020\034\357\037\340\375\003\274\177\000\b\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017@\004\330[\006\000g\000\200\007\230\\\357\037\340!\003<d\000\007\000w\000\200\007\230\\\006\006\007\000\000\000\220\200\b\004g\000\000\0038\\\357\037\340\375\003\274\177\000\007\002\007\003\000\000\020\034\004\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\004\000G\000\200\007\230\\\006\007\367\017\000\200\327[\007\007\367\017@\002\330[\357\037\340\375\003<d\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\004\006\007\000\000\000\220\200\357\037\340!\003\274\177\000\b\bG\000\000\000\020\\\b:\207\000\000\000\340\\\006\b\367\001\000\000)8\357\037\340\375\003\274\177\000\007\000g\000\200\007\230\\\006\000\207\000\200\007\230\\\b\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\006\000\207\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\007\006'\000\300\003\3706\006\006'\000\000\000H8\003\003g\000\000\200\020\\\357\037\340\375\003\274\177\000\005\005w\000\000\b\020\\\007\0007\000\200\007\230\\\003\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017\300\001\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\003\006\007\000\000\000\220\200\007\002\007\001\000\000\020\034\357\037\340\375\003\274\177\000\005\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017\300\002\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\006\006\007\000\000\000\260\200\b\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\004:G\000\000\000\340\\\005\004\367\001\000\000)8\357\037\340\375\003\274\177\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\006\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\004\000W\000\200\007\230\\\005\000g\000\200\007\230\\\006\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\006\005'\000@\003\3706\005\005'\000\000\000H8\005\bW\000\000\200\020\\\357\037\340\375\003\274\177\000\006\007g\000\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\006\004\007\000\000\000\220\200\005\002\307\002\000\000\020\034\357\037\340\375\003\274\177\000\007\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\003\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\003\003g\000\000\002\200Y\357\037\340\375\003\274\177\000\005\002\307\002\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\357\037\340\375\003\274\177\000\017\000\007\000\000\000@\342\005\002\007\003\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\003\004\027\000\000\000\000\034\005\002\007\003\000\000\020\034\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340=\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\000\007\234\377\017@\342\357\037\340\375\003\274\177\000\005\002\307\002\000\000\020\034\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\357\037\340\375\003\274\177\000\003\377G\000\000 Y\\\003\0007\000\200\007\230\\\005\002\367\017\000\200\020\\\357\037\340\375\003\274\177\000\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\357\037\340=\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\005\002\367\017\000\200\020\\\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003\274\177\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\003\360\007\000\000\360\003\001\357\037\340\375\003\274\177\000\005\360\327\211\271\273\003\001\003\004W\000\200\001\200Y\003\3777\000\000\000]\\\357\037\340\375\003\274\177\000\005\360\027\000\000\264\004\001\006\360\007\000\3007\004\001\003\003g\000\200\002\210Y\357\037\340\375\003\274\177\000\005\003\367\007\000\264\f\b\005\377W\000\000 Y\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\360\267\243\212\373\003\001\005\004g\000\200\002\200Y\006\360\007\006W*\003\001\357\037\340\375\003\274\177\000\005\004g\000\200\002\200Y\003\0007\000\200\007\230\\\003\003w\001\000\000H8\357\037\340\375\003<d\000\003\0007\000\200\007\230\\\004\000W\000\200\000\220\\\004\004'\000\000\000\200P\357\037\340\375\003\274\177\000\003\0047\000\000\000h\\\003\003\007\200?\000X8\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\003\007\000\000\030\000\034\004\004\007\000\000\370\007\004\357\037\340\375\003\274\177\000\005\360\367\377\377\037\000\001\007\004W\000\200\003h[\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\000\000\200\n\000\000\220\342\017\000\200\005\000\000@\342\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\@\000\200#\000\000`\342\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340!\003\274\177\000\017\000\007\000\000\000\370\360\004\003G\000\000\000\200P\005\360\007\000\000\370\013\001\357\037\340\375\003\274\177\000\005\003G\000\200\002\200Y\005\377W\000\0000Y\\\005\004W\000\000\002\200Y\357\037\340\375\003\274\177\000\003\000W\000\200\007\230\\\017\000\007\000\000\000\370\360\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\005\002\207\001\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\006\000G\000\200\007\230\\\007\000W\000\200\007\230\\\005\002\207\002\000\000\020\034\357\037\340\375\003\274\177\000\000\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\000\330[\004\000G\000\200\007\230\\\357\037\340!\003<d\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\004:G\000\000\000\340\\\357\037\340\375\003\274\177\000\000\004\367\001\000\000)8\004\000G\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\000\004'\000@\000\3706\005\004'\000\000\000H8\005\006W\000\000\200\020\\\357\037\340\375\003\274\177\000\000\007\007\000\000\b\020\\\005\000W\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017@\000\330[\004\000G\000\200\007\230\\\357\037\340=\003\274\177\000\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\017\000\007\000\000\000\000\343\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\003\027\000\000\000H8\004\004\207\001\000\000(8\005\000\367\017\200\007\230\\\357\037\340\375\003\274\177\000\007\004W\000\200\003j[\005\0007\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\000G\000\200\007\230\\\000\000\2006\000\000\240\342\017\000\000\016\000\000@\342\357\037\340\375\003\274\177\000\003\003\027\000\000\000H8\004\000\367\017\200\007\230\\\007\003G\000\200\003k[\357\037\340!\003\274\177\000\017\000\000\002\000\000@\342\005\005G\000\000\000\200P\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\017\000\007\000\000\000@\343\003\360\007\000\000\370\005\001\004\000\367\017\200\007\230\\\357\037\340!\003\274\177\000\003\0057\000\000\002\200Y\004\003G\000\000\000\200P\005\360\007\000\000\370\013\001\357\037\340\375\003\274\177\000\005\003G\000\200\002\200Y\005\377W\000\0000Y\\\005\004W\000\000\002\200Y\357\037\340\375\003\274\177\000\004\360\007\000\000\370\005\001\006\000\367\017\200\007\230\\\004\005G\000\000\003\200Y\357\037\340\375\003\274\177\000\005\0007\000\200\007\230\\\005\000G\000\200\007\230\\\017\000\007\000\000\000@\343\357\037\340\375\003\274\177\000\006\0047\360\377\377\017\034\007\360\027\000\000\000\000\001\007\006w\000\200\003h[\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\017\000\200\"\000\000@\342\n\003\367\377\377\007\000\004\357\037\340\375\003<d\000\005\n\007\000\000\370#\004\005\000W\000\200\007\230\\\007\005G\000\000\000\200P\357\037\340\375\003\274\177\000\b\360\007\000\000\370\013\001\b\005w\000\000\004\200Y\b\377\207\000\0000Y\\\357\037\340\375\003\274\177\000\005\007\207\000\200\003\210Y\005\000W\000\200\007\230\\\t\005\367\377\377\007\000\004\357\037\340\375\003\274\177\000\t\t\007\000\000\b \004\007\007\207\000\200\003\220Y\007\005w\000\200\003\215X\017\031\340\375\003\274\177\000\005:w\000\000 \340\\\005\000W\000\200\007\230\\\007\006\227\000\000\000G\\\357\037\340\375\003\274\177\000\005\005w\000\000\002G\\\007\3607\000\000\000\000\001\007\007g\000\000\000H\\\357\037\340\375\003\274\177\000\007\007\227\000\000\000G\\\007\007g\000\000\000(\\\006\007\027\000\000\000\000\004\357\037\340\375\003\274\177\000\007\007'\000\000\000\000\004\004\004G\360\377\377\017\034\004\tG\000\000\000(\\\357\037\340\375\003\274\177\000\b\004\027\000\000\000\000\034\t\000\367\017\200\007\230\\\006\006\227\000\200\003Z[\017\031\340\375\003\274\177\000\006:g\000\000 \340\\\006\000g\000\200\007\230\\\t\000\367\017\200\007\230\\\357\037\340!\003\274\177\000\005\005\227\000\200\003Z[\005:W\000\000 \340\\\005\000W\000\200\007\230\\\357\037\340\375\003<d\000\t\000\367\017\200\007\230\\\007\007\227\000\200\003Z[\007:w\000\000 \340\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\005\005w\000\000\002G\\\005\006W\000\000\000G\\\017\031\340\375\003\274\177\000\005:W\000\000 \340\\\005\000W\000\200\007\230\\\005\004\207\000\200\002M[\357\037\340\375\003\274\177\000\004\005\027\000\000\000H8\006\000\367\017\200\007\230\\\007\ng\000\200\003d[\357\037\340\375\003\274\177\000\004\004W\000\000\000\240\\\003\003\007\000\000\000\b\004\003\0047\000\000\002G\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\005\0007\000\200\007\230\\\017\000\007\000\000\000@\343\017\031\340\375\003\274\177\000\005\005G\000\000\000\200P\005\000W\000\200\007\230\\\017\000\007\000\000\000@\343\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\005\000W\000\200\007\230\\\003\000W\000\200\007\230\\\357\037\340\377\000\200\037\000\017\000\007\000\000\000 \343\017\000\207\377\377\017@\342\000\017\007\000\000\000\260P\357\037\340\375\003<d\000\001\000\207\000\200\007\230L\001\001\007\376\377\377\017\034\000\000w\003\000\000\310\360\357\037\340\375\003\274\177\000\007\001\007\000\200\003l[\017\000\200\000\000\000@\342\300\000\020\000\000\000\240\343\357\037\340!\003\274\177\000\000\001\367\017\000\000\020\\\000\n\007\000\000\000\340\\\002\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\367\017\200\007\230\\\000\000'\000\200\007\230\\\004\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\000\007\000\200\007\230\\\004\000G\000\200\007\230\\\002\000\027\000\200\007\230L\357\037\340\375\003\274\177\000\000\000\367\017\200\007\230\\\002\003'\000\000\002G\\\000\004\007\000\000\002G\\\357\037\340!\003\274\177\000\003\360\207\025\000\000\000\001\003\003\007\000\000\000\224\357\003\0007\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\007\025\000\000\000\001\004\004\007\000\000\000\225\357\n\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\013\000W\000\200\007\230\\\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\207\024\000\000\000\001\004\004\007\000\000\000\225\357\006\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000W\000\200\007\230\\\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340!\003\274\177\000\004\360\007\024\000\000\000\001\004\004\007\000\000\000\225\357\b\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000W\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\004\000\247\000\200\007\230\\\005\000\267\000\200\007\230\\\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\006\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\t\000\227\000\200\007\230\\\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\013\002\367\017\000\200\020\\\f\000\367\017\000\b\020\\\013\000\267\000\200\007\230\\\357\037\340\375\003\274\177\000\f\000\307\000\200\007\230\\\n\013\367\017\000\200\327[\013\013\367\017@\006\330[\357\037\340\375\003\274g\000\n\000\247\000\200\007\230\\\013\000\267\000\200\007\230\\\b\n\007\000\000\000\260\240\357\037\340\375\003\274\177\000\t\002\207\000\000\000\020\034\n\000\367\017\000\b\020\\\t\000\227\000\200\007\230\\\357\037\340\375\003\274\177\000\n\000\247\000\200\007\230\\\b\t\367\017\000\200\327[\t\t\367\017@\005\330[\357\037\340\375\003\274g\000\b\000\207\000\200\007\230\\\t\000\227\000\200\007\230\\\006\b\007\000\000\000\260\240\357\037\340\375\003\274\177\000\007\002\007\001\000\000\020\034\b\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\357\037\340\375\003\274\177\000\b\000\207\000\200\007\230\\\006\007\367\017\000\200\327[\007\007\367\017@\004\330[\357\037\340\375\003\274g\000\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\004\006\007\000\000\000\260\240\357\037\340\375\003\274\177\000\005\002\207\001\000\000\020\034\006\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017@\003\330[\357\037\340\375\003\274g\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\003\004\007\000\000\000\220\240\017\031\340\375\003\274\177\000\003\000W\002\000\000\310\360\003\0007\000\200\007\230\\\004\000'\000\200\007\230L\017\031\340!\003\274\177\000\003\003G\000\000\0038\\\004\000\027\002\000\000\310\360\004\000G\000\200\007\230\\\357\037\340\375\003\274\177\000\003\003G\000\000\000\020\\\005\002\307\001\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\357\031\340\375\003\274\177\000\003\004\007\000\000\000\220\240\005\002\307\001\000\000\020\034\003\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\003\004\007\000\000\000\220\200\005\002\207\001\000\000\020\034\006\000\367\017\000\b\020\\\357\037\340\375\003\274\177\000\005\000W\000\200\007\230\\\006\000g\000\200\007\230\\\004\005\367\017\000\200\327[\357\037\340\375\003\274\177\000\005\005\367\017@\003\330[\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\017\031\340\375\003\274\177\000\004\004\007\000\000\000\220\200\007\003G\000\200\003m[\002\000'\000\200\007\230\\\357\037\340\375\003\274\177\000\000\000\007\000\200\007\230\\\017\000\0004\000\000@\342\017\000\007\000\000\000@\342\357\037\340\375\003\274\177\000\005\002\367\017\000\200\020\\\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\003\0007\000\200\007\230\\\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\357\037\340\375\003<d\000\004\000G\000\200\007\230\\\005\000W\000\200\007\230\\\004\004\007\000\000\000\260\200\357\037\340\375\003\274\177\000\007\000G\000\200\007\230\\\006\000W\000\200\007\230\\\005\002\307\001\000\000\020\034\357\037\340\375\003\274\177\000\003\000\367\017\000\b\020\\\005\000W\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\004\005\367\017\000\200\327[\005\005\367\017\300\001\330[\004\000G\000\200\007\230\\\357\037\340!\003<d\000\005\000W\000\200\007\230\\\004\004\007\000\000\000\220\200\004:G\000\000\000\340\\\357\037\340\375\003\274\177\000\003\004\367\001\000\000)8\004\000G\000\200\007\230\\\003\0007\000\200\007\230\\\357\037\340\375\003\274\177\000\003\004'\000\300\001\3706\004\004'\000\000\000H8\007\007G\000\000\200\020\\\357\037\340\375\003\274\177\000\005\0067\000\000\b\020\\\007\000w\000\200\007\230\\\005\000W\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017\300\002\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\005\006\007\000\000\000\220\200\007\002\207\000\000\000\020\034\357\037\340\375\003\274\177\000\b\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017@\004\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\006\006\007\000\000\000\260\200\b\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000w\000\200\007\230\\\007\bG\000\000\200\020\\\b\0067\000\000\b\020\\\357\037\340\375\003\274\177\000\007\000w\000\200\007\230\\\b\000\207\000\200\007\230\\\006\007\367\017\000\200\327[\357\037\340\375\003\274\177\000\007\007\367\017@\004\330[\006\000g\000\200\007\230\\\007\000w\000\200\007\230\\\017\031\340\375\003\274\177\000\006\006\007\000\000\000\220\200\005\005g\000\000\000X\\\007\002\007\001\000\000\020\034\357\037\340\375\003\274\177\000\000\000\367\017\000\b\020\\\007\000w\000\200\007\230\\\000\000\007\000\200\007\230\\\357\037\340\375\003\274\177\000\006\007\367\017\000\200\327[\007\007\367\017@\000\330[\006\000g\000\200\007\230\\\357\037\340!\003\274\177\000\007\000w\000\200\007\230\\\006\006\007\000\000\000\260\200\000\000g\000\200\007\230\\\357\037\340\375\003\274\177\000\006\000w\000\200\007\230\\\004\000G\000\000\200\020\\\000\0067\000\000\b\020\\\357\037\340\375\003\274\177\000\003\000G\000\200\007\230\\\000\000\007\000\200\007\230\\\002\003\367\017\000\200\327[\357\037\340\375\003\274\177\000\003\003\367\017@\000\330[\002\000'\000\200\007\230\\\003\0007\000\200\007\230\\\357\031\340\375\003\274\177\000\005\002\007\000\000\000\220\240\017\000\007\000\000\000@\342\017\000\007\000\000\000\000\343\377\007\000\374\000\200\037\000\017\000\007\377\377\017@\342\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P\340\007\000\374\000\200\037\000\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P\000\017\007\000\000\000\260P__CUDA_FTZ\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000d\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\013\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\244\002\000\000\000\000\000\000\"\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\023\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\310\005\000\000\000\000\000\000\370\001\000\000\000\000\000\000\002\000\000\000\021\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\007\000\000\000\000\000\000\344\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000O\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\244\b\000\000\000\000\000\000\240\000\000\000\000\000\000\000\003\000\000\000\016\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000D\t\000\000\000\000\000\000|\000\000\000\000\000\000\000\003\000\000\000\017\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000|\001\000\000\000\000\000p\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\t\000\000\000\000\000\000\000\001\000\000\000\000\000\000\003\000\000\000\020\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\000\000p\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\n\000\000\000\000\000\000|\000\000\000\000\000\000\000\003\000\000\000\021\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000U\002\000\000\013\000\000p\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\013\000\000\000\000\000\000\330\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\254\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\f\000\000\000\000\000\000`\001\000\000\000\000\000\000\000\000\000\000\016\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0006\001\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000x\r\000\000\000\000\000\000\\\001\000\000\000\000\000\000\000\000\000\000\017\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\302\001\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\324\016\000\000\000\000\000\000`\001\000\000\000\000\000\000\000\000\000\000\020\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0006\002\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0004\020\000\000\000\000\000\000\\\001\000\000\000\000\000\000\000\000\000\000\021\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0002\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\240\021\000\000\000\000\000\000@\017\000\000\000\000\000\000\003\000\000\000\021\000\000\016 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\321\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\340 \000\000\000\000\000\000\300\f\000\000\000\000\000\000\003\000\000\000\022\000\000\r \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000]\001\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\240-\000\000\000\000\000\000@\026\000\000\000\000\000\000\003\000\000\000\023\000\000\016 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\351\001\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\340C\000\000\000\000\000\000\000\t\000\000\000\000\000\000\003\000\000\000\024\000\000\r \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\234\000\000\000\001\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\340L\000\000\000\000\000\000\013\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\221\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\353L\000\000\000\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\006\000\000\000\005\000\000\000\360Q\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\250\000\000\000\000\000\000\000\250\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\005\000\000\000\030\f\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\310@\000\000\000\000\000\000\310@\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\006\000\000\000\340L\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\013\000\000\000\000\000\000\000\016\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\001\001H\000\000\000\200\t\000\000\000\000\000\000y\t\000\000@\000\000\000\004\000\007\0002\000\000\000\000\000\000\000\000\000\000\000\021 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\303!\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360 \n\n\n\n.version 7.4\n.target sm_50\n.address_size 64.\000\377\021global .align 1 .b8 blockIdx[1];\"\000\b?Dim\"\000\007othreadE\000\n\360\003_$_str[11] = {95, \004\000P67, 8\b\000C8, 6\024\000\360\00070, 84, 90, 0};\345\000\375\034isible .entry _Z6vecAddPfS_S_i(\n.param .u64\036\000\021_\034\000?_0,&\000\021\0371&\000\022\0272&\000/32&\000\005\2463\n)\n{\n.loc\f\001\0228\f\001\021_\025\000\240_depot0[325\001\313reg .b64 %SP\017\000\024L\020\000\245pred %p<2>\"\000u32 %r<9\021\000\020f\021\000If<4>D\000\343rd<18>;\n\nmov.uV\000\033,\211\000b;\ncvta\261\000\004%\000\023,\200\000\"ld\362\000\001\361\000o%r1, [\367\000\005\030].\000\002}\000\0373/\000\007\0372/\000\000\0372/\000\007\0371/\000\000\017\215\000\b#0]\325\000#to\226\002\0045\000 4,\006\000\0233\037\000\n\034\000\0215\034\000\0374;\000\005\0216\037\000\0372;\000\002\0217\034\000\0376;\000\005\0218\037\000\0371;\000\002\0219\034\000Q8;\nst\023\000q[%SP+0]\026\000\0329\026\000\0228\026\000\0327\026\000\"16\027\000\0225\027\000\"32\027\000!24\027\000\"1;\375\001\001\300\001\2702, %ctaid.x\027\000c3, %nt\026\000qul.lo.s\031\000#4,5\000(r30\000\000)\001\003/\0003add,\000$6,1\000\f\211\000\002\267\000\0216\302\001\002A\000%7,\033\000\007\026\000%8,\272\000\222;\nsetp.ge]\0002p1,6\000\362\016%r8;\n@%p1 bra LBB0_2;\nbra.uni\020\00021;\n\b\000\021:Z\000\002e\001410,Y\001\001q\000\002\263\0008d11\211\0004shl7\003412, \000\0232\345\000\003\031\000$3,P\000\000\007\000\0212N\000\002}\003\001L\000\000\"\000(];z\000$4,\275\001\nI\000$5,\037\000\fI\000\0232I\000\02352\000#rn\031\000\"3,g\0009%f2c\000%6,\n\002\nd\000$7, \000\003d\000!stK\000\001_\000\0207\227\001+f3?\001\0232?\001\2572:\nret;\n\n}\212\005\001\34312forward_pass}\003\r\222\005\017&\000\003\016\232\005\017.\000\020\0371.\000\032\r\252\005\017.\000\n\0373.\000\032\0374\340\005\023Y1[56]\215\005\017\340\005\017\0343\340\005,22\341\005926>F\000\"rd%\000\017\342\005\t\0371\342\005\030\002\204\005\017\001\001\013\036]\030\006\0176\000\017\017 \006\002\0177\000\017\017(\006\002\0177\000\017\0170\006\002\0177\000\007\0178\006\266\03788\006\002/169\006\002/249\006\002$329\006\t\026\000\002D\000\03428\006\017O\006\000\0374O\006\005#5,5\000(r40\000\0376O\006\001$7,1\000\0326\211\000\0224\034\007\03779\006\000(40O\006%9,\320\000\017O\006\002\001]\001\0329O\00691_6\020\005\0231O\006\0201O\006\006\262\000\000\377\005\0330\227\000\003o\007\f\027\000\0258\027\000\tW\000\0232W\000\0222\246\006\002V\000%1,5\000\007\310\000\000\214\006\004\203\001\f\311\000#2,8\000\000'\000\001\313\000\0262\313\000\0335t\000\0233t\000/3:\240\006\005\006t\000\0375S\001\002(16\213\000\t\334\001\002l\006\0211\262\001\0301\354\007(18\326\000\006\341\001619,7\000\0218\343\002\000\270\006\003\036\000\003 \007*19\227\007$6,\034\000\0372\352\006\002\0234\361\006\bN\007$2,\360\006\030;\370\000\03085\007\f}\000\0249\226\000\007}\000\0232\364\007\0349}\000421,R\000\001'\000\b}\000\0223}\000821]\026\000%4,5\002V;\nfma\343\007\0202\367\000\001\264\000\002=\000\000-\000\007\267\007\005i\002,f2\336\001\0234\336\001\0274R\002/20|\001\003\002\262\000\000\036\000\013\346\003\003\257\002/21\257\002\004+5:\360\b\004\305\0006neg\276\b:%f1\264\000&0]\265\b\005\313\b\003\032\000\002\n\004\002\026\000\2014, 0f3F0\001\000\b\031\000\0215\031\000{BBB989D3\001\"6,P\000\001&\0003%f4\316\001\021a\200\000\003$\000\000\037\002(f6R\000\0208R\00004B4j\000\0301\031\000\0219\031\000237C\204\000\001k\000\023m\034\000\001\367\t\002M\000\000\"\002+f8\241\t\001\212\003\001&\00010fC\\\000(7F\034\001\001W\n)f1p\000\000R\n\000\365\000kB8AA3B\334\000%14\335\000\0221\006\000\0312<\000\002\030\001|2A57060<\000\006\031\001!15\006\000\003v\005\002]\b\003P\000\0230\344\002\002\024\000#4,\032\000'23,\000\023f\246\003\364\0024;\nex2.approx.ftzj\000\000\372\0022f16\023\004\006\201\000&9,\035\000\0337+\001\000}\002\002#\000\000\375\000\0238\362\0017rcp#\000\001N\001\003P\003\f\311\013\0202v\002\024l\321\002\006\312\013\001\261\004\007\266\003\001s\001\017\312\013\023\005\262\002\001\305\013\001|\000\f\b\003\0236\b\003\0376\036\013\013\\3backe\b\f\260\020\017&\000\003\016\036\013\017.\000\020\0371.\000\032\016\036\013\017.\000\t\017\320\020\024!2[z\001\007\233\n\017\320\0201912>E\000\000\357\n\0374\357\n\f\0372\321\020\036\016\000\001\017\331\020\022\0167\000\017\341\020\022\0167\000\017\351\020\022\0167\000\017\361\020\377\34392_2\224\004\0232\242\n\0372\361\020\2646sub\002\006\"3,g\000\001<\b\t?\006\0254\334\006\0273\271\005\005\276\f)f4\223\000\0376\r\001\003\0377\r\001\005\003\277\002\0357\r\001$9,P\000\000\007\000\0278\304\000\0235\304\000\0319\321\b\0326\337\006\t\335\000\"7,!\000;%f59\007\"8,Q\000;%f7\335\000\001\372\002\030fx\000\t\346\f\005\026\000,10\251\013\013\244\b#9,\375\007\006&\001(20\022\013\006'\001/21'\001\004\001\246\n\002 \000\n4\002\001\271\n\0051\013\0322[\007423]\005\t\t\307\002\0232\307\002\017y\022\f\2570calc_gradw\022\007\017$\000\000\016W\007\017,\000\016\0371,\000\030\016S\007\017,\000\007\0373,\000\b\017m\022$\0363}\007\017m\022 \0340~\007)5>E\000\000}\007\0376}\007\f\0373l\022\037\016\376\000\017j\022\017\0164\000\017\257\007\020\0165\000\017f\022\020\0165\000\017d\022\020\0165\000\017\251\007\377\004\t\026\000\0378a\022t/32a\022\006/32a\022\001/28\277\007\003\017a\022\001\0333a\022\0233\277\007\0373\277\007\004/16\300\007\003/32\300\007&\006\030\001\f\307\022\004\370\f\r\306\022%36\027\000\t\274\000\0232\274\000\017\306\022\002/36\306\022\002/24\306\022\031\0333\306\022\0233\306\022\0373\306\022\022\0303\213\000\033c\344\021\0225\354\016\nG\001$6,\034\000\013G\001\0377a\022\006/1,`\022\n\0310\222\000\03044\002\t\274\002\004\220\000\004j\017\017u\023\000#2]\224\000\017r\023\002\f\340\000\003`\023\t\340\000\003\223\007\017\304\022\033\002\340\000)21\340\000(22\242\002\013\\\001$23\225\000\b|\000$4,\034\000\013|\000(5,O\b\0274|\000\0233|\000\000\032\n\t\335\020\"4,z\001\001\237\000\n\005\n\001/\000\003\004\n\t:\002\0234:\002\b\"\023(18#\002\001\225\000\b\236\024\0358\"\023\004\013\003\0379\013\003\004*5:\030\000\0236\030\000\0032\020P\n\n}\n\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_5, 23729

	.type	__cuda_fatbin_wrapper,@object   # @__cuda_fatbin_wrapper
	.section	.nvFatBinSegment,"aw",@progbits
	.p2align	3
__cuda_fatbin_wrapper:
	.long	1180844977                      # 0x466243b1
	.long	1                               # 0x1
	.quad	.L__unnamed_5
	.quad	0
	.size	__cuda_fatbin_wrapper, 24

	.type	__cuda_gpubin_handle,@object    # @__cuda_gpubin_handle
	.local	__cuda_gpubin_handle
	.comm	__cuda_gpubin_handle,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_simple_nn.cu
	.quad	__cuda_module_ctor
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"Ubuntu clang version 14.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __cxx_global_var_init
	.addrsig_sym __cxa_atexit
	.addrsig_sym MPI_Type_contiguous
	.addrsig_sym MPI_Wait
	.addrsig_sym MPI_Start
	.addrsig_sym MPI_Grequest_complete
	.addrsig_sym MPI_Send
	.addrsig_sym _ZNK3MPI8DatatypecvP15ompi_datatype_tEv
	.addrsig_sym MPI_Scan
	.addrsig_sym _ZNK3MPI2OpcvP9ompi_op_tEv
	.addrsig_sym MPI_Comm_dup
	.addrsig_sym _Znwm
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZdlPv
	.addrsig_sym MPI_Group_size
	.addrsig_sym MPI_Errhandler_free
	.addrsig_sym MPI_Get_count
	.addrsig_sym MPI_Info_delete
	.addrsig_sym MPI_Win_set_errhandler
	.addrsig_sym _ZNK3MPI10ErrhandlercvP17ompi_errhandler_tEv
	.addrsig_sym _Z21__device_stub__vecAddPfS_S_i
	.addrsig_sym __cudaPopCallConfiguration
	.addrsig_sym cudaLaunchKernel
	.addrsig_sym _Z27__device_stub__forward_passPfS_S_ii
	.addrsig_sym _Z28__device_stub__backward_passPfS_S_i
	.addrsig_sym _Z25__device_stub__calc_gradsPfS_S_ii
	.addrsig_sym MPI_Init
	.addrsig_sym printf
	.addrsig_sym exit
	.addrsig_sym MPI_Comm_rank
	.addrsig_sym MPI_Comm_size
	.addrsig_sym _ZL11getHostNamePci
	.addrsig_sym _ZL11getHostHashPKc
	.addrsig_sym MPI_Allgather
	.addrsig_sym MPI_Bcast
	.addrsig_sym cudaStreamCreate
	.addrsig_sym cudaGetErrorString
	.addrsig_sym ncclCommInitRank
	.addrsig_sym ncclGetErrorString
	.addrsig_sym malloc
	.addrsig_sym rand
	.addrsig_sym srand
	.addrsig_sym time
	.addrsig_sym _ZL10cudaMallocIfE9cudaErrorPPT_m
	.addrsig_sym cudaMemcpy
	.addrsig_sym cudaMemset
	.addrsig_sym __cudaPushCallConfiguration
	.addrsig_sym cudaGetLastError
	.addrsig_sym ncclAllReduce
	.addrsig_sym cudaStreamSynchronize
	.addrsig_sym cudaFree
	.addrsig_sym free
	.addrsig_sym ncclCommDestroy
	.addrsig_sym MPI_Finalize
	.addrsig_sym _ZN3MPI14Is_initializedEv
	.addrsig_sym MPI_Topo_test
	.addrsig_sym MPI_Initialized
	.addrsig_sym gethostname
	.addrsig_sym MPI_Type_vector
	.addrsig_sym MPI_Type_indexed
	.addrsig_sym MPI_Type_create_hindexed
	.addrsig_sym MPI_Type_create_hvector
	.addrsig_sym MPI_Type_create_indexed_block
	.addrsig_sym MPI_Type_create_resized
	.addrsig_sym MPI_Type_size
	.addrsig_sym MPI_Type_get_extent
	.addrsig_sym MPI_Type_get_true_extent
	.addrsig_sym MPI_Type_commit
	.addrsig_sym MPI_Pack
	.addrsig_sym _ZNK3MPI9Comm_NullcvP19ompi_communicator_tEv
	.addrsig_sym MPI_Unpack
	.addrsig_sym MPI_Pack_size
	.addrsig_sym MPI_Pack_external
	.addrsig_sym MPI_Pack_external_size
	.addrsig_sym MPI_Unpack_external
	.addrsig_sym MPI_Type_create_subarray
	.addrsig_sym MPI_Type_create_darray
	.addrsig_sym MPI_Type_dup
	.addrsig_sym MPI_Type_delete_attr
	.addrsig_sym MPI_Type_get_attr
	.addrsig_sym _Znam
	.addrsig_sym MPI_Type_get_contents
	.addrsig_sym _ZN3MPI8DatatypeaSERKP15ompi_datatype_t
	.addrsig_sym _ZdaPv
	.addrsig_sym MPI_Type_get_envelope
	.addrsig_sym MPI_Type_get_name
	.addrsig_sym MPI_Type_set_attr
	.addrsig_sym MPI_Type_set_name
	.addrsig_sym MPI_Test_cancelled
	.addrsig_sym MPI_Get_elements
	.addrsig_sym MPI_Status_set_elements
	.addrsig_sym MPI_Status_set_cancelled
	.addrsig_sym MPI_Test
	.addrsig_sym MPI_Request_free
	.addrsig_sym MPI_Cancel
	.addrsig_sym MPI_Request_get_status
	.addrsig_sym _ZN3MPI6StatusaSERK20ompi_status_public_t
	.addrsig_sym MPI_Group_rank
	.addrsig_sym MPI_Group_incl
	.addrsig_sym MPI_Group_excl
	.addrsig_sym MPI_Group_range_incl
	.addrsig_sym MPI_Group_range_excl
	.addrsig_sym MPI_Group_free
	.addrsig_sym MPI_Recv
	.addrsig_sym MPI_Bsend
	.addrsig_sym MPI_Ssend
	.addrsig_sym MPI_Rsend
	.addrsig_sym MPI_Isend
	.addrsig_sym MPI_Ibsend
	.addrsig_sym MPI_Issend
	.addrsig_sym MPI_Irsend
	.addrsig_sym MPI_Irecv
	.addrsig_sym MPI_Iprobe
	.addrsig_sym MPI_Probe
	.addrsig_sym MPI_Send_init
	.addrsig_sym MPI_Bsend_init
	.addrsig_sym MPI_Ssend_init
	.addrsig_sym MPI_Rsend_init
	.addrsig_sym MPI_Recv_init
	.addrsig_sym MPI_Sendrecv
	.addrsig_sym MPI_Sendrecv_replace
	.addrsig_sym MPI_Comm_group
	.addrsig_sym MPI_Comm_free
	.addrsig_sym MPI_Comm_test_inter
	.addrsig_sym MPI_Barrier
	.addrsig_sym MPI_Gather
	.addrsig_sym MPI_Gatherv
	.addrsig_sym MPI_Scatter
	.addrsig_sym MPI_Scatterv
	.addrsig_sym MPI_Allgatherv
	.addrsig_sym MPI_Alltoall
	.addrsig_sym MPI_Alltoallv
	.addrsig_sym MPI_Alltoallw
	.addrsig_sym MPI_Reduce
	.addrsig_sym MPI_Allreduce
	.addrsig_sym MPI_Reduce_scatter
	.addrsig_sym MPI_Comm_disconnect
	.addrsig_sym MPI_Comm_get_name
	.addrsig_sym MPI_Comm_set_name
	.addrsig_sym MPI_Abort
	.addrsig_sym MPI_Comm_set_errhandler
	.addrsig_sym MPI_Comm_get_errhandler
	.addrsig_sym MPI_Comm_set_attr
	.addrsig_sym MPI_Comm_get_attr
	.addrsig_sym MPI_Comm_delete_attr
	.addrsig_sym MPI_Win_get_errhandler
	.addrsig_sym MPI_Accumulate
	.addrsig_sym MPI_Win_complete
	.addrsig_sym MPI_Win_fence
	.addrsig_sym MPI_Get
	.addrsig_sym MPI_Win_get_group
	.addrsig_sym MPI_Win_lock
	.addrsig_sym MPI_Win_post
	.addrsig_sym _ZNK3MPI5GroupcvP12ompi_group_tEv
	.addrsig_sym MPI_Put
	.addrsig_sym MPI_Win_start
	.addrsig_sym MPI_Win_test
	.addrsig_sym MPI_Win_unlock
	.addrsig_sym MPI_Win_wait
	.addrsig_sym MPI_Win_call_errhandler
	.addrsig_sym MPI_Win_delete_attr
	.addrsig_sym MPI_Win_get_name
	.addrsig_sym MPI_Win_set_attr
	.addrsig_sym MPI_Win_set_name
	.addrsig_sym MPI_Exscan
	.addrsig_sym MPI_Comm_create
	.addrsig_sym MPI_Comm_split
	.addrsig_sym MPI_Intercomm_create
	.addrsig_sym MPI_Cart_create
	.addrsig_sym MPI_Graph_create
	.addrsig_sym MPI_Comm_accept
	.addrsig_sym _ZNK3MPI4InfocvP11ompi_info_tEv
	.addrsig_sym MPI_Comm_connect
	.addrsig_sym MPI_Comm_spawn
	.addrsig_sym _ZN3MPI9Intracomm24convert_info_to_mpi_infoEiPKNS_4InfoE
	.addrsig_sym MPI_Comm_spawn_multiple
	.addrsig_sym MPI_Cartdim_get
	.addrsig_sym MPI_Cart_get
	.addrsig_sym MPI_Cart_rank
	.addrsig_sym MPI_Cart_coords
	.addrsig_sym MPI_Cart_shift
	.addrsig_sym MPI_Cart_sub
	.addrsig_sym MPI_Cart_map
	.addrsig_sym MPI_Graphdims_get
	.addrsig_sym MPI_Graph_get
	.addrsig_sym MPI_Graph_neighbors_count
	.addrsig_sym MPI_Graph_neighbors
	.addrsig_sym MPI_Graph_map
	.addrsig_sym MPI_Comm_remote_size
	.addrsig_sym MPI_Comm_remote_group
	.addrsig_sym MPI_Intercomm_merge
	.addrsig_sym MPI_Info_dup
	.addrsig_sym MPI_Info_free
	.addrsig_sym MPI_Info_get
	.addrsig_sym MPI_Info_get_nkeys
	.addrsig_sym MPI_Info_get_nthkey
	.addrsig_sym MPI_Info_get_valuelen
	.addrsig_sym MPI_Info_set
	.addrsig_sym MPI_Op_create
	.addrsig_sym ompi_mpi_cxx_op_intercept
	.addrsig_sym ompi_op_set_cxx_callback
	.addrsig_sym MPI_Op_free
	.addrsig_sym MPI_Reduce_local
	.addrsig_sym MPI_Op_commutative
	.addrsig_sym cudaMalloc
	.addrsig_sym _GLOBAL__sub_I_simple_nn.cu
	.addrsig_sym __cuda_register_globals
	.addrsig_sym __cudaRegisterFunction
	.addrsig_sym __cudaRegisterFatBinary
	.addrsig_sym __cuda_module_ctor
	.addrsig_sym __cudaRegisterFatBinaryEnd
	.addrsig_sym __cudaUnregisterFatBinary
	.addrsig_sym __cuda_module_dtor
	.addrsig_sym atexit
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym ompi_mpi_comm_world
	.addrsig_sym ompi_mpi_datatype_null
	.addrsig_sym ompi_mpi_byte
	.addrsig_sym _ZTVN10__cxxabiv117__class_type_infoE
	.addrsig_sym _ZTSN3MPI8DatatypeE
	.addrsig_sym _ZTIN3MPI8DatatypeE
	.addrsig_sym _ZTSN3MPI6StatusE
	.addrsig_sym _ZTIN3MPI6StatusE
	.addrsig_sym _ZTSN3MPI7RequestE
	.addrsig_sym _ZTIN3MPI7RequestE
	.addrsig_sym _ZTVN10__cxxabiv120__si_class_type_infoE
	.addrsig_sym _ZTSN3MPI8PrequestE
	.addrsig_sym _ZTIN3MPI8PrequestE
	.addrsig_sym _ZTSN3MPI8GrequestE
	.addrsig_sym _ZTIN3MPI8GrequestE
	.addrsig_sym _ZTSN3MPI5GroupE
	.addrsig_sym _ZTIN3MPI5GroupE
	.addrsig_sym _ZTSN3MPI4CommE
	.addrsig_sym _ZTSN3MPI9Comm_NullE
	.addrsig_sym _ZTIN3MPI9Comm_NullE
	.addrsig_sym _ZTIN3MPI4CommE
	.addrsig_sym _ZTSN3MPI3WinE
	.addrsig_sym _ZTIN3MPI3WinE
	.addrsig_sym _ZTSN3MPI10ErrhandlerE
	.addrsig_sym _ZTIN3MPI10ErrhandlerE
	.addrsig_sym _ZTSN3MPI9IntracommE
	.addrsig_sym _ZTIN3MPI9IntracommE
	.addrsig_sym _ZTSN3MPI8CartcommE
	.addrsig_sym _ZTIN3MPI8CartcommE
	.addrsig_sym _ZTSN3MPI9GraphcommE
	.addrsig_sym _ZTIN3MPI9GraphcommE
	.addrsig_sym _ZTSN3MPI9IntercommE
	.addrsig_sym _ZTIN3MPI9IntercommE
	.addrsig_sym _ZTSN3MPI4InfoE
	.addrsig_sym _ZTIN3MPI4InfoE
	.addrsig_sym _ZTSN3MPI2OpE
	.addrsig_sym _ZTIN3MPI2OpE
	.addrsig_sym ompi_mpi_comm_null
	.addrsig_sym .L__unnamed_5
	.addrsig_sym __cuda_fatbin_wrapper
	.addrsig_sym __cuda_gpubin_handle
