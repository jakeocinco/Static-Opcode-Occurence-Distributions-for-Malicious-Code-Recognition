from functools import reduce
import os
import numpy as np
from PIL import Image

OP_CODE_DICT = {
    'base': {
        'benign': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull', 'incl',
                   'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js', 'leal',
                   'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb', 'orl', 'pushl', 'rep', 'sbbb', 'sbbl',
                   'sete', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'infected': ['adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull', 'incl',
                     'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jne', 'js', 'leal',
                     'leave', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep', 'retl', 'sarl',
                     'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'union': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull',
                  'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jne', 'jns',
                  'js', 'leal', 'leave', 'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb', 'orl', 'popl',
                  'pushl', 'rep', 'retl', 'sarl', 'sbbb', 'sbbl', 'sete', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl',
                  'xchgl', 'xorb', 'xorl'],
        'intersection': ['adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull', 'incl', 'int3',
                         'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'js', 'leal', 'movb',
                         'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'pushl', 'rep', 'sbbl', 'shrl', 'subb', 'subl',
                         'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'disjoint': ['adcb', 'calll', 'cmpw', 'jmpl', 'jns', 'leave', 'lock', 'movzwl', 'popl', 'retl', 'sarl', 'sbbb',
                     'sete', 'shll'],
        'ratio': ['addb', 'addl', 'andb', 'andl', 'bsrw ', 'bswapw', 'btcw', 'btrw', 'btsw', 'cmovnsq', 'cmpb', 'cmpl',
                  'cvtsi2sdl', 'data16', 'decl', 'ds', 'es', 'fs', 'gs', 'imull', 'incl', 'iretq', 'ja', 'jae', 'jb', 'jbe',
                  'je', 'jl', 'jle', 'jmp', 'jne', 'js', 'leal', 'lretq', 'movb', 'movl', 'movw', 'nop', 'orl', 'rex64',
                  'sbbl', 'scasq', 'sgdtq', 'shldw', 'ss', 'subl', 'testb', 'testl', 'xorb', 'xorl'],
        'ratio_a25': ['adcs', 'add', 'adds', 'adr', 'and', 'ands', 'asrs', 'b', 'beq', 'bgt', 'bhs', 'bl', 'ble', 'bls',
                      'blx', 'bne', 'bx', 'cbnz', 'cbz', 'cmp', 'dmb', 'ldm', 'ldr', 'ldrb', 'ldrh', 'ldrsb', 'ldrsh',
                      'lgdtq', 'lidtq', 'lsls', 'lsrs', 'mov', 'movs', 'movzww', 'mvn', 'mvns', 'orrs', 'pop', 'shldw',
                      'shrdw', 'sidtq', 'stm', 'str', 'strb', 'strd', 'strh', 'sub', 'subs', 'tst', 'udf'],
        'ratio_a75': ['adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull',
                      'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js',
                      'leal', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep', 'retl', 'sbbb',
                      'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'malware_cluster': ['add', 'addb', 'addl', 'addpd', 'addps', 'addq', 'adds', 'addsd', 'addss', 'addw', 'call',
                            'calll', 'callq', 'callw', 'cmp', 'cmpb', 'cmpl', 'cmppd', 'cmpps', 'cmpq', 'cmpsb', 'cmpsl',
                            'cmpsq', 'cmpss', 'cmpsw', 'cmpw', 'jmp', 'jmpl', 'jmpq', 'jmpw', 'lea', 'leal', 'leaq',
                            'leave', 'leaw', 'mov', 'movb', 'movd', 'movi', 'movk', 'movl', 'movq', 'movsb', 'movsd',
                            'movsl', 'movsq', 'movss', 'movsw', 'movw', 'pop', 'popal', 'popaw', 'popfl', 'popfq', 'popfw',
                            'popl', 'popq', 'popw', 'push', 'pushal', 'pushaw', 'pushfl', 'pushfq', 'pushfw', 'pushl',
                            'pushq', 'pushw', 'shll', 'sub', 'subb', 'subl', 'subpd', 'subps', 'subq', 'subs', 'subsd',
                            'subss', 'subw', 'test', 'testb', 'testl', 'testq', 'testw', 'xor', 'xorb', 'xorl', 'xorpd',
                            'xorps', 'xorq', 'xorw'],
        'mov_cluster': ['mov', 'movb', 'movl', 'movsb', 'movsbl', 'movsl', 'movswl', 'movw', 'movzbl', 'movzwl'],
        'common_mov': ['mov', 'movb', 'movl', 'movsb', 'movsbl', 'movsl', 'movswl', 'movw', 'movzbl', 'movzwl'],
        'common_cmp': ['cmpl', 'cmpb', 'cmpw', 'cmpsb', 'cmpsl'],
        'common_add': ['addl', 'addb', 'addw', 'faddl', 'fadds', 'fiadds', 'fiaddl'], # bad,
        'common_cmp_mov': ['cmpl', 'cmpb', 'cmpw', 'cmpsb', 'cmpsl', 'mov', 'movl', 'movb', 'movw', 'movzwl', 'movzbl',
                           'movsl', 'movsb', 'movsbl', 'movswl'],
        'lea_shll_test_cluster': ['lea', 'leal', 'leaq', 'leave', 'leaw', 'shll', 'test', 'testb', 'testl', 'testq', 'testw'],
        'common_test': ['test', 'testb', 'testl', 'testq', 'testw'],
        'mov_test_cluster': ['mov', 'movb', 'movd', 'movi', 'movk', 'movl', 'movq', 'movsb', 'movsd', 'movsl', 'movsq',
                             'movss', 'movsw', 'movw', 'test', 'testb', 'testl', 'testq', 'testw'],
        'common_cluster': ['add', 'addb', 'addl', 'addpd', 'addps', 'addq', 'adds', 'addsd', 'addss', 'addsubps',
                           'addw', 'and', 'andb', 'andl', 'andnl', 'andnpd', 'andnps', 'andnq', 'andpd', 'andps', 'andq', 'ands', 'andw', 'call', 'calll', 'callq', 'callw', 'ccmp', 'cmovael', 'cmovaeq', 'cmovaew', 'cmoval', 'cmovaq', 'cmovaw', 'cmovbel', 'cmovbeq', 'cmovbew', 'cmovbl', 'cmovbq', 'cmovbw', 'cmovel', 'cmoveq', 'cmovew', 'cmovgel', 'cmovgeq', 'cmovgew', 'cmovgl', 'cmovgq', 'cmovgw', 'cmovlel', 'cmovleq', 'cmovlew', 'cmovll', 'cmovlq', 'cmovlw', 'cmovnel', 'cmovneq', 'cmovnew', 'cmovnol', 'cmovnoq', 'cmovnpl', 'cmovnpq', 'cmovnpw', 'cmovnsl', 'cmovnsq', 'cmovnsw', 'cmovol', 'cmovoq', 'cmovpl', 'cmovpq', 'cmovsl', 'cmovsq', 'cmovsw', 'cmp', 'cmpb', 'cmpeqpd', 'cmpeqps', 'cmpeqsd', 'cmpeqss', 'cmpl', 'cmplepd', 'cmpleps', 'cmplesd', 'cmpless', 'cmpltpd', 'cmpltps', 'cmpltsd', 'cmpltss', 'cmpneqpd', 'cmpneqps', 'cmpneqsd', 'cmpneqss', 'cmpnlepd', 'cmpnleps', 'cmpnlesd', 'cmpnless', 'cmpnltpd', 'cmpnltps', 'cmpnltsd', 'cmpordpd', 'cmpordps', 'cmpordsd', 'cmpordss', 'cmppd', 'cmpps', 'cmpq', 'cmpsb', 'cmpsl', 'cmpsq', 'cmpss', 'cmpsw', 'cmpunordpd', 'cmpunordps', 'cmpunordsd', 'cmpunordss', 'cmpw', 'cmpxchg16b', 'cmpxchg8b', 'cmpxchgb', 'cmpxchgl', 'cmpxchgq', 'cmpxchgw', 'cset', 'csetm', 'div', 'divb', 'divl', 'divpd', 'divps', 'divq', 'divsd', 'divss', 'divw', 'eor', 'fadd', 'faddl', 'faddp', 'fadds', 'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb', 'fcmovnbe', 'fcmovne', 'fcmovnu', 'fcmovu', 'fdiv', 'fdivl', 'fdivp', 'fdivr', 'fdivrl', 'fdivrp', 'fdivrs', 'fdivs', 'fiaddl', 'fiadds', 'fidivl', 'fidivrl', 'fidivrs', 'fidivs', 'fimull', 'fimuls', 'fisubl', 'fisubrl', 'fisubrs', 'fisubs', 'fmadd', 'fmov', 'fmul', 'fmull', 'fmulp', 'fmuls', 'fnmsub', 'fnsave', 'frstor', 'fsub', 'fsubl', 'fsubp', 'fsubr', 'fsubrl', 'fsubrp', 'fsubrs', 'fsubs', 'fxrstor', 'fxsave', 'hsubps', 'idivb', 'idivl', 'idivq', 'idivw', 'imulb', 'imull', 'imulq', 'imulw', 'iretl', 'iretq', 'iretw', 'jmp', 'jmpl', 'jmpq', 'jmpw', 'kandb', 'kmovb', 'kmovd', 'kmovw', 'kxnorb', 'kxnorw', 'lcalll', 'lcallq', 'lcallw', 'lea', 'leal', 'leaq', 'leave', 'leaw', 'ljmpl', 'ljmpq', 'ljmpw', 'loop', 'loope', 'loopne', 'lretl', 'lretq', 'lretw', 'madd', 'maskmovdqu', 'maskmovq', 'max', 'maxpd', 'maxps', 'maxsd', 'maxss', 'min', 'minpd', 'minps', 'minsd', 'minss', 'monitor', 'monitorx', 'mov', 'movabsb', 'movabsl', 'movabsq', 'movabsw', 'movapd', 'movaps', 'movb', 'movbel', 'movbeq', 'movd', 'movddup', 'movdiri', 'movdq2q', 'movdqa', 'movdqu', 'movhlps', 'movhpd', 'movhps', 'movi', 'movk', 'movl', 'movlhps', 'movlpd', 'movlps', 'movmskpd', 'movmskps', 'movntdq', 'movntdqa', 'movntil', 'movntiq', 'movntps', 'movntq', 'movq', 'movq2dq', 'movsb', 'movsbl', 'movsbq', 'movsbw', 'movsd', 'movshdup', 'movsl', 'movsldup', 'movslq', 'movsq', 'movss', 'movsw', 'movswl', 'movswq', 'movupd', 'movups', 'movw', 'movzbl', 'movzbq', 'movzbw', 'movzwl', 'movzwq', 'msub', 'mul', 'mulb', 'mull', 'mulpd', 'mulps', 'mulq', 'mulsd', 'mulss', 'mulw', 'mulxq', 'or', 'orb', 'orl', 'orn', 'orpd', 'orps', 'orq', 'orr', 'orw', 'paddb', 'paddd', 'paddq', 'paddsb', 'paddsw', 'paddusb', 'paddusw', 'paddw', 'pand', 'pandn', 'pclmulqdq', 'pcmpeqb', 'pcmpeqd', 'pcmpeqq', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw', 'pcmpistri', 'pfadd', 'pfcmpge', 'pfcmpgt', 'pfmax', 'pfmin', 'pfmul', 'pfsub', 'pfsubr', 'phaddd', 'phaddsw', 'phaddw', 'phminposuw', 'phsubd', 'phsubw', 'pmaddubsw', 'pmaddwd', 'pmaxsb', 'pmaxsd', 'pmaxsw', 'pmaxub', 'pmaxud', 'pmaxuw', 'pminsb', 'pminsd', 'pminsw', 'pminub', 'pminud', 'pminuw', 'pmovmskb', 'pmovsxbd', 'pmovsxbq', 'pmovsxbw', 'pmovsxdq', 'pmovsxwd', 'pmovsxwq', 'pmovzxbd', 'pmovzxbq', 'pmovzxbw', 'pmovzxdq', 'pmovzxwd', 'pmovzxwq', 'pmuldq', 'pmulhrsw', 'pmulhrw', 'pmulhuw', 'pmulhw', 'pmulld', 'pmullw', 'pmuludq', 'pop', 'popal', 'popaw', 'popcntl', 'popfl', 'popfq', 'popfw', 'popl', 'popq', 'popw', 'por', 'psubb', 'psubd', 'psubq', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'psubw', 'ptest', 'push', 'pushal', 'pushaw', 'pushfl', 'pushfq', 'pushfw', 'pushl', 'pushq', 'pushw', 'pxor', 'rdrandl', 'rdrandq', 'ret', 'retl', 'retq', 'retw', 'ror', 'rorb', 'rorl', 'rorq', 'rorw', 'rorxl', 'rorxq', 'save', 'sdiv', 'set', 'seta', 'setae', 'setb', 'setbe', 'sete', 'setg', 'setge', 'setl', 'setle', 'setne', 'setno', 'setnp', 'setns', 'seto', 'setp', 'sets', 'sha', 'sha1msg1', 'sha1msg2', 'sha1nexte', 'sha1rnds4', 'sha256msg1', 'sha256msg2', 'sha256rnds2', 'shll', 'smull', 'sub', 'subb', 'subl', 'subpd', 'subps', 'subq', 'subs', 'subsd', 'subss', 'subw', 'sys', 'syscall', 'sysenter', 'sysexitl', 'sysexitq', 'sysretl', 'sysretq', 'test', 'testb', 'testl', 'testq', 'testw', 'uaddlv', 'udiv', 'umaxv', 'uminv', 'umull', 'vaddpd', 'vaddps', 'vaddsd', 'vaddss', 'vaddsubpd', 'vaddsubps', 'vandnpd', 'vandnps', 'vandpd', 'vandps', 'vcmpeq_uqsd', 'vcmpeq_uqss', 'vcmpeqpd', 'vcmpeqps', 'vcmpeqsd', 'vcmpeqss', 'vcmpfalse_ossd', 'vcmpge_oqsd', 'vcmpgt_oqsd', 'vcmpgtps', 'vcmpgtss', 'vcmplepd', 'vcmpleps', 'vcmplesd', 'vcmplt_oqsd', 'vcmpneq_ussd', 'vcmpnge_uqpd', 'vcmpnge_uqss', 'vcmpngt_uqps', 'vcmpngtps', 'vcmpnltpd', 'vcmpnltsd', 'vcmpordps', 'vcmppd', 'vcmpps', 'vcmpsd', 'vcmpss', 'vdivpd', 'vdivps', 'vdivsd', 'vdivss', 'vfmadd132ps', 'vfmadd132sd', 'vfmadd132ss', 'vfmadd213ps', 'vfmadd213sd', 'vfmadd213ss', 'vfmadd231sd', 'vfmadd231ss', 'vfmaddsub132ps', 'vfmaddsubpd', 'vfmaddsubps', 'vfmsub132sd', 'vfmsub213sd', 'vfmsub213ss', 'vfmsub231ps', 'vfmsubadd132ps', 'vfmsubadd231pd', 'vfmsubadd231ps', 'vfmsubaddpd', 'vfmsubpd', 'vfmsubps', 'vfmsubss', 'vfnmadd132ps', 'vfnmadd132sd', 'vfnmadd213ps', 'vfnmadd213sd', 'vfnmadd231sd', 'vfnmaddps', 'vfnmaddss', 'vfnmsub132ps', 'vfnmsub231sd', 'vfnmsubpd', 'vfnmsubps', 'vfnmsubsd', 'vhaddpd', 'vhaddps', 'vhsubpd', 'vhsubps', 'vmaskmovpd', 'vmaskmovps', 'vmaxpd', 'vmaxps', 'vmaxsd', 'vmaxss', 'vmcall', 'vminpd', 'vminps', 'vminsd', 'vminss', 'vmmcall', 'vmovapd', 'vmovaps', 'vmovd', 'vmovddup', 'vmovdqa', 'vmovdqa32', 'vmovdqa64', 'vmovdqu', 'vmovdqu32', 'vmovdqu64', 'vmovdqu8', 'vmovhpd', 'vmovhps', 'vmovlpd', 'vmovlps', 'vmovmskpd', 'vmovntdq', 'vmovntps', 'vmovq', 'vmovsd', 'vmovshdup', 'vmovsldup', 'vmovss', 'vmovupd', 'vmovups', 'vmsave', 'vmulpd', 'vmulps', 'vmulsd', 'vmulss', 'vorpd', 'vorps', 'vpaddb', 'vpaddd', 'vpaddq', 'vpaddsb', 'vpaddsw', 'vpaddusb', 'vpaddusw', 'vpaddw', 'vpand', 'vpandd', 'vpandn', 'vpandq', 'vpclmulqdq', 'vpcmpeqb', 'vpcmpeqd', 'vpcmpeqq', 'vpcmpeqw', 'vpcmpgtb', 'vpcmpgtd', 'vpcmpgtq', 'vpcmpgtw', 'vpcmpltw', 'vpcmpnleub', 'vphaddd', 'vphaddsw', 'vphaddw', 'vphminposuw', 'vphsubw', 'vpmadd52huq', 'vpmadd52luq', 'vpmaddubsw', 'vpmaddwd', 'vpmaskmovd', 'vpmaskmovq', 'vpmaxsb', 'vpmaxsd', 'vpmaxsw', 'vpmaxub', 'vpmaxud', 'vpmaxuw', 'vpminsb', 'vpminsd', 'vpminsq', 'vpminsw', 'vpminub', 'vpminuq', 'vpminuw', 'vpmovdw', 'vpmovmskb', 'vpmovsxbw', 'vpmovsxdq', 'vpmovsxwd', 'vpmovzxbd', 'vpmovzxbq', 'vpmovzxbw', 'vpmovzxwd', 'vpmovzxwq', 'vpmuldq', 'vpmulhrsw', 'vpmulhuw', 'vpmulhw', 'vpmulld', 'vpmullw', 'vpmuludq', 'vpor', 'vporq', 'vpshaw', 'vpsubb', 'vpsubd', 'vpsubq', 'vpsubsb', 'vpsubsw', 'vpsubusb', 'vpsubusw', 'vpsubw', 'vptest', 'vpxor', 'vpxord', 'vpxorq', 'vsubpd', 'vsubps', 'vsubsd', 'vsubss', 'vxorpd', 'vxorps', 'xabort', 'xaddb', 'xaddl', 'xaddq', 'xor', 'xorb', 'xorl', 'xorpd', 'xorps', 'xorq', 'xorw', 'xrelease', 'xrstor', 'xrstors', 'xsave', 'xsave64', 'xsavec', 'xsaveopt', 'xsaveopt64', 'xsaves', 'xsha1', 'xsha256', 'xstorerng', 'xtest'],
        'mull_cluster': ['fimull', 'fmull', 'imull', 'mull', 'pmulld', 'pmullw', 'smull', 'umull', 'vpmulld', 'vpmullw']
    },
    'jump': {
        'benign': ['adcb', 'adcl', 'addb', 'andb', 'andl', 'cltd', 'cmpl', 'cmpw', 'decl', 'imull', 'incl', 'ja', 'jae',
                   'jb', 'jbe', 'je', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js', 'movb', 'movl', 'movzwl', 'negl',
                   'nop', 'orb', 'orl', 'sbbl', 'sete', 'shrl', 'subb', 'subl', 'testl', 'xchgl', 'xorl'],
        'infected': ['adcl', 'addb', 'andb', 'andl', 'calll', 'cltd', 'cmpl', 'decl', 'imull', 'incl', 'ja', 'jae',
                     'jb', 'jbe', 'je', 'jge', 'jl', 'jle', 'jmp', 'jne', 'js', 'movb', 'movl', 'negl', 'nop', 'orb',
                     'orl', 'popl', 'retl', 'sarl', 'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testl', 'xchgl', 'xorl'],
        'union': ['adcb', 'adcl', 'addb', 'andb', 'andl', 'calll', 'cltd', 'cmpl', 'cmpw', 'decl', 'imull', 'incl',
                  'ja', 'jae', 'jb', 'jbe', 'je', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js', 'movb', 'movl',
                  'movzwl', 'negl', 'nop', 'orb', 'orl', 'popl', 'retl', 'sarl', 'sbbl', 'sete', 'shll', 'shrl', 'subb',
                  'subl', 'testl', 'xchgl', 'xorl'],
        'intersection': ['adcl', 'addb', 'andb', 'andl', 'cltd', 'cmpl', 'decl', 'imull', 'incl', 'ja', 'jae', 'jb',
                         'jbe', 'je', 'jge', 'jl', 'jle', 'jmp', 'jne', 'js', 'movb', 'movl', 'negl', 'nop', 'orb',
                         'orl', 'sbbl', 'shrl', 'subb', 'subl', 'testl', 'xchgl', 'xorl'],
        'disjoint': ['adcb', 'calll', 'cmpw', 'jns', 'movzwl', 'popl', 'retl', 'sarl', 'sete', 'shll'],
        'ratio_a75': ['adcl', 'addb', 'andb', 'andl', 'calll', 'cltd', 'cmpl', 'cmpw', 'decl', 'imull', 'incl', 'ja',
                      'jae', 'jb', 'jbe', 'je', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js', 'movb', 'movl', 'negl',
                      'nop', 'orb', 'orl', 'popl', 'retl', 'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testl', 'xchgl',
                      'xorl'],
        'ratio': ['addb', 'andb', 'andl', 'bsrw ', 'bswapw', 'btcw', 'btrw', 'btsw', 'cmovnsq', 'cmpl', 'cvtsi2sdl',
                  'data16', 'decl', 'ds', 'es', 'fs', 'gs', 'imull', 'incl', 'iretq', 'ja', 'jae', 'jb', 'jbe', 'je',
                  'jl', 'jle', 'jmp', 'jne', 'js', 'lretq', 'movb', 'movl', 'nop', 'orl', 'rex64', 'sbbl', 'scasq',
                  'sgdtq', 'shldw', 'ss', 'subl', 'testl', 'xorl']
    },
    'share': {
        'benign': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull',
                   'incl', 'int3', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js',
                   'leal', 'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orl', 'pushl', 'rep', 'sbbb',
                   'sbbl', 'sete', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorl'],
        'infected': ['addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull', 'incl', 'int3',
                     'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jne', 'js', 'leal',
                     'leave', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep', 'retl',
                     'sarl', 'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'union': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull',
                  'incl', 'int3', 'ja', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jne', 'jns', 'js',
                  'leal', 'leave', 'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb', 'orl', 'popl',
                  'pushl', 'rep', 'sarl', 'sbbb', 'sbbl', 'sete', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl',
                  'xchgl', 'xorb', 'xorl'],
        'intersection': ['adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull', 'incl',
                         'int3', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jne', 'js', 'leal', 'movb',
                         'movl', 'movw', 'negl', 'nop', 'orl', 'pushl', 'rep', 'sbbl', 'shrl', 'subb', 'subl', 'testb',
                         'testl', 'xchgl', 'xorl'],
        'disjoint': ['adcb', 'calll', 'cmpw', 'jns', 'leave', 'lock', 'movzwl', 'popl', 'retl', 'sarl', 'sbbb', 'sete',
                     'shll'],
        'ratio_a75': ['addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull', 'incl',
                      'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js',
                      'leal', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep', 'retl',
                      'sbbb', 'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'ratio': ['addb', 'addl', 'andb', 'andl', 'bsrw ', 'bswapw', 'btcw', 'btrw', 'btsw', 'cmovnsq', 'cmpb', 'cmpl',
                  'cvtsi2sdl', 'data16', 'decl', 'ds', 'es', 'fs', 'gs', 'imull', 'incl', 'iretq', 'ja', 'jae', 'jb',
                  'jbe', 'je', 'jl', 'jle', 'jmp', 'jne', 'js', 'leal', 'lretq', 'movb', 'movl', 'nop', 'orl', 'rex64',
                  'sbbl', 'scasq', 'sgdtq', 'shldw', 'ss', 'subl', 'testb', 'testl', 'xorb', 'xorl']
    },
    'cumulative_share': {
        'benign': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull',
                   'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js',
                   'leal', 'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb', 'orl', 'pushl', 'rep',
                   'sbbb', 'sbbl', 'sete', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'infected': ['adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull', 'incl',
                     'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jne', 'js',
                     'leal', 'leave', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep',
                     'retl', 'sarl', 'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'union': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl',
                  'imull', 'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl',
                  'jne', 'jns', 'js', 'leal', 'leave', 'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb',
                  'orl', 'popl', 'pushl', 'rep', 'retl', 'sarl', 'sbbb', 'sbbl', 'sete', 'shll', 'shrl', 'subb', 'subl',
                  'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'intersection': ['adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'decl', 'imull', 'incl',
                         'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'js', 'leal',
                         'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'pushl', 'rep', 'sbbl', 'shrl', 'subb',
                         'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'disjoint': ['adcb', 'calll', 'cmpw', 'jmpl', 'jns', 'leave', 'lock', 'movzwl', 'popl', 'retl', 'sarl', 'sbbb',
                     'sete', 'shll'],
        'ratio_a75': ['adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull',
                      'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns',
                      'js', 'leal', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep', 'retl',
                      'sbbb', 'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'ratio': ['addb', 'addl', 'andb', 'andl', 'bsrw ', 'bswapw', 'btcw', 'btrw', 'btsw', 'cmovnsq', 'cmpb', 'cmpl',
                  'cvtsi2sdl', 'data16', 'decl', 'ds', 'es', 'fs', 'gs', 'imull', 'incl', 'iretq', 'ja', 'jae', 'jb',
                  'jbe', 'je', 'jl', 'jle', 'jmp', 'jne', 'js', 'leal', 'lretq', 'movb', 'movl', 'movw', 'nop', 'orl',
                  'rex64', 'sbbl', 'scasq', 'sgdtq', 'shldw', 'ss', 'subl', 'testb', 'testl', 'xorb', 'xorl']
    },
    'inverse_jump': {
        'benign': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'imull', 'incl',
                   'int3', 'ja', 'jae', 'jb', 'jbe', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js', 'leal',
                   'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb', 'orl', 'pushl', 'rep', 'sbbb',
                   'sbbl', 'sete', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xorb', 'xorl'],
        'infected': ['adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'imull', 'incl', 'int3',
                     'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jne', 'js', 'leal',
                     'leave', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'rep', 'retl', 'sarl',
                     'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'union': ['adcb', 'adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl',
                  'imull', 'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl',
                  'jne', 'jns', 'js', 'leal', 'leave', 'lock', 'movb', 'movl', 'movw', 'movzwl', 'negl', 'nop', 'orb',
                  'orl', 'popl', 'pushl', 'rep', 'retl', 'sarl', 'sbbl', 'sete', 'shll', 'shrl', 'subb', 'subl',
                  'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'intersection': ['adcl', 'addb', 'addl', 'andb', 'andl', 'cltd', 'cmpb', 'cmpl', 'imull', 'incl', 'int3', 'ja',
                         'jae', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'js', 'leal', 'movb', 'movl', 'movw',
                         'negl', 'nop', 'orb', 'orl', 'pushl', 'rep', 'sbbl', 'shrl', 'subb', 'subl', 'testb', 'testl',
                         'xorb', 'xorl'],
        'disjoint': ['adcb', 'calll', 'cmpw', 'jmpl', 'jns', 'leave', 'lock', 'movzwl', 'popl', 'retl', 'sarl', 'sbbb',
                     'sete', 'shll'],
        'ratio_a75': ['adcl', 'addb', 'addl', 'andb', 'andl', 'calll', 'cltd', 'cmpb', 'cmpl', 'cmpw', 'decl', 'imull',
                      'incl', 'int3', 'ja', 'jae', 'jb', 'jbe', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js',
                      'leal', 'movb', 'movl', 'movw', 'negl', 'nop', 'orb', 'orl', 'popl', 'pushl', 'rep', 'retl',
                      'sbbl', 'shll', 'shrl', 'subb', 'subl', 'testb', 'testl', 'xchgl', 'xorb', 'xorl'],
        'ratio': ['addb', 'addl', 'andb', 'andl', 'bsrw ', 'bswapw', 'btcw', 'btrw', 'btsw', 'cmovnsq', 'cmpb', 'cmpl',
                  'cvtsi2sdl', 'data16', 'decl', 'ds', 'es', 'fs', 'gs', 'imull', 'incl', 'iretq', 'ja', 'jae', 'jb',
                  'jbe', 'je', 'jl', 'jle', 'jmp', 'jne', 'js', 'leal', 'lretq', 'movb', 'movl', 'movw', 'nop', 'orl',
                  'rex64', 'sbbl', 'scasq', 'sgdtq', 'shldw', 'ss', 'subl', 'testb', 'testl', 'xorb', 'xorl']
    }
}

OP_CODE_CLUSTER = {
    'malware_cluster': {'add': 'add', 'addb': 'add', 'addl': 'add', 'addpd': 'add', 'addps': 'add', 'addq': 'add',
                        'adds': 'add', 'addsd': 'add', 'addss': 'add', 'addw': 'add', 'call': 'call', 'calll': 'call',
                        'callq': 'call', 'callw': 'call', 'cmp': 'cmp', 'cmpb': 'cmp', 'cmpl': 'cmp', 'cmppd': 'cmp',
                        'cmpps': 'cmp', 'cmpq': 'cmp', 'cmpsb': 'cmp', 'cmpsl': 'cmp', 'cmpsq': 'cmp', 'cmpss': 'cmp',
                        'cmpsw': 'cmp', 'cmpw': 'cmp', 'jmp': 'jmp', 'jmpl': 'jmp', 'jmpq': 'jmp', 'jmpw': 'jmp',
                        'lea': 'lea', 'leal': 'lea', 'leaq': 'lea', 'leave': 'lea', 'leaw': 'lea', 'mov': 'mov',
                        'movb': 'mov', 'movd': 'mov', 'movi': 'mov', 'movk': 'mov', 'movl': 'mov', 'movq': 'mov',
                        'movsb': 'mov', 'movsd': 'mov', 'movsl': 'mov', 'movsq': 'mov', 'movss': 'mov', 'movsw':'mov',
                        'movw': 'mov', 'pop': 'pop', 'popal': 'pop', 'popaw': 'pop', 'popfl': 'pop',
                        'popfq': 'pop', 'popfw': 'pop', 'popl': 'pop', 'popq': 'pop', 'popw': 'pop', 'push': 'push',
                        'pushal': 'push', 'pushaw': 'push', 'pushfl': 'push', 'pushfq': 'push', 'pushfw': 'push',
                        'pushl': 'push', 'pushq': 'push', 'pushw': 'push', 'shll': 'shll', 'sub': 'sub', 'subb': 'sub',
                        'subl': 'sub', 'subpd': 'sub', 'subps': 'sub', 'subq': 'sub', 'subs': 'sub', 'subsd': 'sub',
                        'subss': 'sub', 'subw': 'sub', 'test': 'test', 'testb': 'test', 'testl': 'test',
                        'testq': 'test', 'testw': 'test', 'xor': 'xor', 'xorb': 'xor', 'xorl': 'xor', 'xorpd': 'xor',
                        'xorps': 'xor', 'xorq': 'xor', 'xorw': 'xor'},
    'mov_cluster': {'mov': 'mov', 'movb': 'mov', 'movl': 'mov', 'movsb': 'mov', 'movsbl': 'mov', 'movsl': 'mov',
                    'movswl': 'mov', 'movw': 'mov', 'movzbl': 'mov', 'movzwl': 'mov'},
    'lea_shll_test_cluster': {'lea': 'lea', 'leal': 'lea', 'leaq': 'lea', 'leave': 'lea', 'leaw': 'lea', 'shll': 'shll',
                              'test': 'test', 'testb': 'test', 'testl': 'test', 'testq': 'test', 'testw': 'test'},
    'mov_test_cluster': {'mov': 'mov', 'movb': 'mov', 'movd': 'mov', 'movi': 'mov', 'movk': 'mov', 'movl': 'mov',
                         'movq': 'mov', 'movsb': 'mov', 'movsd': 'mov', 'movsl': 'mov', 'movsq': 'mov', 'movss': 'mov',
                         'movsw': 'mov', 'movw': 'mov', 'test': 'test', 'testb': 'test', 'testl': 'test',
                         'testq': 'test', 'testw': 'test'},
    'common_cluster': {'add': 'add', 'addb': 'add', 'addl': 'add', 'addpd': 'add', 'addps': 'add', 'addq': 'add',
                       'adds': 'add', 'addsd': 'add', 'addss': 'add', 'addsubps': 'sub', 'addw': 'add', 'and': 'and', 'andb': 'and', 'andl': 'and', 'andnl': 'and', 'andnpd': 'and', 'andnps': 'and', 'andnq': 'and', 'andpd': 'and', 'andps': 'and', 'andq': 'and', 'ands': 'and', 'andw': 'and', 'call': 'call', 'calll': 'call', 'callq': 'call', 'callw': 'call', 'ccmp': 'cmp', 'cmovael': 'mov', 'cmovaeq': 'mov', 'cmovaew': 'mov', 'cmoval': 'mov', 'cmovaq': 'mov', 'cmovaw': 'mov', 'cmovbel': 'mov', 'cmovbeq': 'mov', 'cmovbew': 'mov', 'cmovbl': 'mov', 'cmovbq': 'mov', 'cmovbw': 'mov', 'cmovel': 'mov', 'cmoveq': 'mov', 'cmovew': 'mov', 'cmovgel': 'mov', 'cmovgeq': 'mov', 'cmovgew': 'mov', 'cmovgl': 'mov', 'cmovgq': 'mov', 'cmovgw': 'mov', 'cmovlel': 'mov', 'cmovleq': 'mov', 'cmovlew': 'mov', 'cmovll': 'mov', 'cmovlq': 'mov', 'cmovlw': 'mov', 'cmovnel': 'mov', 'cmovneq': 'mov', 'cmovnew': 'mov', 'cmovnol': 'mov', 'cmovnoq': 'mov', 'cmovnpl': 'mov', 'cmovnpq': 'mov', 'cmovnpw': 'mov', 'cmovnsl': 'mov', 'cmovnsq': 'mov', 'cmovnsw': 'mov', 'cmovol': 'mov', 'cmovoq': 'mov', 'cmovpl': 'mov', 'cmovpq': 'mov', 'cmovsl': 'mov', 'cmovsq': 'mov', 'cmovsw': 'mov', 'cmp': 'cmp', 'cmpb': 'cmp', 'cmpeqpd': 'cmp', 'cmpeqps': 'cmp', 'cmpeqsd': 'cmp', 'cmpeqss': 'cmp', 'cmpl': 'cmp', 'cmplepd': 'cmp', 'cmpleps': 'cmp', 'cmplesd': 'cmp', 'cmpless': 'cmp', 'cmpltpd': 'cmp', 'cmpltps': 'cmp', 'cmpltsd': 'cmp', 'cmpltss': 'cmp', 'cmpneqpd': 'cmp', 'cmpneqps': 'cmp', 'cmpneqsd': 'cmp', 'cmpneqss': 'cmp', 'cmpnlepd': 'cmp', 'cmpnleps': 'cmp', 'cmpnlesd': 'cmp', 'cmpnless': 'cmp', 'cmpnltpd': 'cmp', 'cmpnltps': 'cmp', 'cmpnltsd': 'cmp', 'cmpordpd': 'or', 'cmpordps': 'or', 'cmpordsd': 'or', 'cmpordss': 'or', 'cmppd': 'cmp', 'cmpps': 'cmp', 'cmpq': 'cmp', 'cmpsb': 'cmp', 'cmpsl': 'cmp', 'cmpsq': 'cmp', 'cmpss': 'cmp', 'cmpsw': 'cmp', 'cmpunordpd': 'or', 'cmpunordps': 'or', 'cmpunordsd': 'or', 'cmpunordss': 'or', 'cmpw': 'cmp', 'cmpxchg16b': 'cmp', 'cmpxchg8b': 'cmp', 'cmpxchgb': 'cmp', 'cmpxchgl': 'cmp', 'cmpxchgq': 'cmp', 'cmpxchgw': 'cmp', 'cset': 'set', 'csetm': 'set', 'div': 'div', 'divb': 'div', 'divl': 'div', 'divpd': 'div', 'divps': 'div', 'divq': 'div', 'divsd': 'div', 'divss': 'div', 'divw': 'div', 'eor': 'or', 'fadd': 'add', 'faddl': 'add', 'faddp': 'add', 'fadds': 'add', 'fcmovb': 'mov', 'fcmovbe': 'mov', 'fcmove': 'mov', 'fcmovnb': 'mov', 'fcmovnbe': 'mov', 'fcmovne': 'mov', 'fcmovnu': 'mov', 'fcmovu': 'mov', 'fdiv': 'div', 'fdivl': 'div', 'fdivp': 'div', 'fdivr': 'div', 'fdivrl': 'div', 'fdivrp': 'div', 'fdivrs': 'div', 'fdivs': 'div', 'fiaddl': 'add', 'fiadds': 'add', 'fidivl': 'div', 'fidivrl': 'div', 'fidivrs': 'div', 'fidivs': 'div', 'fimull': 'mul', 'fimuls': 'mul', 'fisubl': 'sub', 'fisubrl': 'sub', 'fisubrs': 'sub', 'fisubs': 'sub', 'fmadd': 'add', 'fmov': 'mov', 'fmul': 'mul', 'fmull': 'mul', 'fmulp': 'mul', 'fmuls': 'mul', 'fnmsub': 'sub', 'fnsave': 'save', 'frstor': 'or', 'fsub': 'sub', 'fsubl': 'sub', 'fsubp': 'sub', 'fsubr': 'sub', 'fsubrl': 'sub', 'fsubrp': 'sub', 'fsubrs': 'sub', 'fsubs': 'sub', 'fxrstor': 'or', 'fxsave': 'save', 'hsubps': 'sub', 'idivb': 'div', 'idivl': 'div', 'idivq': 'div', 'idivw': 'div', 'imulb': 'mul', 'imull': 'mul', 'imulq': 'mul', 'imulw': 'mul', 'iretl': 'ret', 'iretq': 'ret', 'iretw': 'ret', 'jmp': 'jmp', 'jmpl': 'jmp', 'jmpq': 'jmp', 'jmpw': 'jmp', 'kandb': 'and', 'kmovb': 'mov', 'kmovd': 'mov', 'kmovw': 'mov', 'kxnorb': 'or', 'kxnorw': 'or', 'lcalll': 'call', 'lcallq': 'call', 'lcallw': 'call', 'lea': 'lea', 'leal': 'lea', 'leaq': 'lea', 'leave': 'lea', 'leaw': 'lea', 'ljmpl': 'jmp', 'ljmpq': 'jmp', 'ljmpw': 'jmp', 'loop': 'loop', 'loope': 'loop', 'loopne': 'loop', 'lretl': 'ret', 'lretq': 'ret', 'lretw': 'ret', 'madd': 'add', 'maskmovdqu': 'mov', 'maskmovq': 'mov', 'max': 'max', 'maxpd': 'max', 'maxps': 'max', 'maxsd': 'max', 'maxss': 'max', 'min': 'min', 'minpd': 'min', 'minps': 'min', 'minsd': 'min', 'minss': 'min', 'monitor': 'or', 'monitorx': 'or', 'mov': 'mov', 'movabsb': 'mov', 'movabsl': 'mov', 'movabsq': 'mov', 'movabsw': 'mov', 'movapd': 'mov', 'movaps': 'mov', 'movb': 'mov', 'movbel': 'mov', 'movbeq': 'mov', 'movd': 'mov', 'movddup': 'mov', 'movdiri': 'mov', 'movdq2q': 'mov', 'movdqa': 'mov', 'movdqu': 'mov', 'movhlps': 'mov', 'movhpd': 'mov', 'movhps': 'mov', 'movi': 'mov', 'movk': 'mov', 'movl': 'mov', 'movlhps': 'mov', 'movlpd': 'mov', 'movlps': 'mov', 'movmskpd': 'mov', 'movmskps': 'mov', 'movntdq': 'mov', 'movntdqa': 'mov', 'movntil': 'mov', 'movntiq': 'mov', 'movntps': 'mov', 'movntq': 'mov', 'movq': 'mov', 'movq2dq': 'mov', 'movsb': 'mov', 'movsbl': 'mov', 'movsbq': 'mov', 'movsbw': 'mov', 'movsd': 'mov', 'movshdup': 'mov', 'movsl': 'mov', 'movsldup': 'mov', 'movslq': 'mov', 'movsq': 'mov', 'movss': 'mov', 'movsw': 'mov', 'movswl': 'mov', 'movswq': 'mov', 'movupd': 'mov', 'movups': 'mov', 'movw': 'mov', 'movzbl': 'mov', 'movzbq': 'mov', 'movzbw': 'mov', 'movzwl': 'mov', 'movzwq': 'mov', 'msub': 'sub', 'mul': 'mul', 'mulb': 'mul', 'mull': 'mul', 'mulpd': 'mul', 'mulps': 'mul', 'mulq': 'mul', 'mulsd': 'mul', 'mulss': 'mul', 'mulw': 'mul', 'mulxq': 'mul', 'or': 'or', 'orb': 'or', 'orl': 'or', 'orn': 'or', 'orpd': 'or', 'orps': 'or', 'orq': 'or', 'orr': 'or', 'orw': 'or', 'paddb': 'add', 'paddd': 'add', 'paddq': 'add', 'paddsb': 'add', 'paddsw': 'add', 'paddusb': 'add', 'paddusw': 'add', 'paddw': 'add', 'pand': 'and', 'pandn': 'and', 'pclmulqdq': 'mul', 'pcmpeqb': 'cmp', 'pcmpeqd': 'cmp', 'pcmpeqq': 'cmp', 'pcmpeqw': 'cmp', 'pcmpgtb': 'cmp', 'pcmpgtd': 'cmp', 'pcmpgtw': 'cmp', 'pcmpistri': 'cmp', 'pfadd': 'add', 'pfcmpge': 'cmp', 'pfcmpgt': 'cmp', 'pfmax': 'max', 'pfmin': 'min', 'pfmul': 'mul', 'pfsub': 'sub', 'pfsubr': 'sub', 'phaddd': 'add', 'phaddsw': 'add', 'phaddw': 'add', 'phminposuw': 'min', 'phsubd': 'sub', 'phsubw': 'sub', 'pmaddubsw': 'add', 'pmaddwd': 'add', 'pmaxsb': 'max', 'pmaxsd': 'max', 'pmaxsw': 'max', 'pmaxub': 'max', 'pmaxud': 'max', 'pmaxuw': 'max', 'pminsb': 'min', 'pminsd': 'min', 'pminsw': 'min', 'pminub': 'min', 'pminud': 'min', 'pminuw': 'min', 'pmovmskb': 'mov', 'pmovsxbd': 'mov', 'pmovsxbq': 'mov', 'pmovsxbw': 'mov', 'pmovsxdq': 'mov', 'pmovsxwd': 'mov', 'pmovsxwq': 'mov', 'pmovzxbd': 'mov', 'pmovzxbq': 'mov', 'pmovzxbw': 'mov', 'pmovzxdq': 'mov', 'pmovzxwd': 'mov', 'pmovzxwq': 'mov', 'pmuldq': 'mul', 'pmulhrsw': 'mul', 'pmulhrw': 'mul', 'pmulhuw': 'mul', 'pmulhw': 'mul', 'pmulld': 'mul', 'pmullw': 'mul', 'pmuludq': 'mul', 'pop': 'pop', 'popal': 'pop', 'popaw': 'pop', 'popcntl': 'pop', 'popfl': 'pop', 'popfq': 'pop', 'popfw': 'pop', 'popl': 'pop', 'popq': 'pop', 'popw': 'pop', 'por': 'or', 'psubb': 'sub', 'psubd': 'sub', 'psubq': 'sub', 'psubsb': 'sub', 'psubsw': 'sub', 'psubusb': 'sub', 'psubusw': 'sub', 'psubw': 'sub', 'ptest': 'test', 'push': 'push', 'pushal': 'sha', 'pushaw': 'sha', 'pushfl': 'push', 'pushfq': 'push', 'pushfw': 'push', 'pushl': 'push', 'pushq': 'push', 'pushw': 'push', 'pxor': 'xor', 'rdrandl': 'and', 'rdrandq': 'and', 'ret': 'ret', 'retl': 'ret', 'retq': 'ret', 'retw': 'ret', 'ror': 'or', 'rorb': 'or', 'rorl': 'or', 'rorq': 'or', 'rorw': 'or', 'rorxl': 'or', 'rorxq': 'or', 'save': 'save', 'sdiv': 'div', 'set': 'set', 'seta': 'set', 'setae': 'set', 'setb': 'set', 'setbe': 'set', 'sete': 'set', 'setg': 'set', 'setge': 'set', 'setl': 'set', 'setle': 'set', 'setne': 'set', 'setno': 'set', 'setnp': 'set', 'setns': 'set', 'seto': 'set', 'setp': 'set', 'sets': 'set', 'sha': 'sha', 'sha1msg1': 'sha', 'sha1msg2': 'sha', 'sha1nexte': 'sha', 'sha1rnds4': 'sha', 'sha256msg1': 'sha', 'sha256msg2': 'sha', 'sha256rnds2': 'sha', 'shll': 'shll', 'smull': 'mul', 'sub': 'sub', 'subb': 'sub', 'subl': 'sub', 'subpd': 'sub', 'subps': 'sub', 'subq': 'sub', 'subs': 'sub', 'subsd': 'sub', 'subss': 'sub', 'subw': 'sub', 'sys': 'sys', 'syscall': 'sys', 'sysenter': 'sys', 'sysexitl': 'sys', 'sysexitq': 'sys', 'sysretl': 'sys', 'sysretq': 'sys', 'test': 'test', 'testb': 'test', 'testl': 'test', 'testq': 'test', 'testw': 'test', 'uaddlv': 'add', 'udiv': 'div', 'umaxv': 'max', 'uminv': 'min', 'umull': 'mul', 'vaddpd': 'add', 'vaddps': 'add', 'vaddsd': 'add', 'vaddss': 'add', 'vaddsubpd': 'sub', 'vaddsubps': 'sub', 'vandnpd': 'and', 'vandnps': 'and', 'vandpd': 'and', 'vandps': 'and', 'vcmpeq_uqsd': 'cmp', 'vcmpeq_uqss': 'cmp', 'vcmpeqpd': 'cmp', 'vcmpeqps': 'cmp', 'vcmpeqsd': 'cmp', 'vcmpeqss': 'cmp', 'vcmpfalse_ossd': 'cmp', 'vcmpge_oqsd': 'cmp', 'vcmpgt_oqsd': 'cmp', 'vcmpgtps': 'cmp', 'vcmpgtss': 'cmp', 'vcmplepd': 'cmp', 'vcmpleps': 'cmp', 'vcmplesd': 'cmp', 'vcmplt_oqsd': 'cmp', 'vcmpneq_ussd': 'cmp', 'vcmpnge_uqpd': 'cmp', 'vcmpnge_uqss': 'cmp', 'vcmpngt_uqps': 'cmp', 'vcmpngtps': 'cmp', 'vcmpnltpd': 'cmp', 'vcmpnltsd': 'cmp', 'vcmpordps': 'or', 'vcmppd': 'cmp', 'vcmpps': 'cmp', 'vcmpsd': 'cmp', 'vcmpss': 'cmp', 'vdivpd': 'div', 'vdivps': 'div', 'vdivsd': 'div', 'vdivss': 'div', 'vfmadd132ps': 'add', 'vfmadd132sd': 'add', 'vfmadd132ss': 'add', 'vfmadd213ps': 'add', 'vfmadd213sd': 'add', 'vfmadd213ss': 'add', 'vfmadd231sd': 'add', 'vfmadd231ss': 'add', 'vfmaddsub132ps': 'sub', 'vfmaddsubpd': 'sub', 'vfmaddsubps': 'sub', 'vfmsub132sd': 'sub', 'vfmsub213sd': 'sub', 'vfmsub213ss': 'sub', 'vfmsub231ps': 'sub', 'vfmsubadd132ps': 'sub', 'vfmsubadd231pd': 'sub', 'vfmsubadd231ps': 'sub', 'vfmsubaddpd': 'sub', 'vfmsubpd': 'sub', 'vfmsubps': 'sub', 'vfmsubss': 'sub', 'vfnmadd132ps': 'add', 'vfnmadd132sd': 'add', 'vfnmadd213ps': 'add', 'vfnmadd213sd': 'add', 'vfnmadd231sd': 'add', 'vfnmaddps': 'add', 'vfnmaddss': 'add', 'vfnmsub132ps': 'sub', 'vfnmsub231sd': 'sub', 'vfnmsubpd': 'sub', 'vfnmsubps': 'sub', 'vfnmsubsd': 'sub', 'vhaddpd': 'add', 'vhaddps': 'add', 'vhsubpd': 'sub', 'vhsubps': 'sub', 'vmaskmovpd': 'mov', 'vmaskmovps': 'mov', 'vmaxpd': 'max', 'vmaxps': 'max', 'vmaxsd': 'max', 'vmaxss': 'max', 'vmcall': 'call', 'vminpd': 'min', 'vminps': 'min', 'vminsd': 'min', 'vminss': 'min', 'vmmcall': 'call', 'vmovapd': 'mov', 'vmovaps': 'mov', 'vmovd': 'mov', 'vmovddup': 'mov', 'vmovdqa': 'mov', 'vmovdqa32': 'mov', 'vmovdqa64': 'mov', 'vmovdqu': 'mov', 'vmovdqu32': 'mov', 'vmovdqu64': 'mov', 'vmovdqu8': 'mov', 'vmovhpd': 'mov', 'vmovhps': 'mov', 'vmovlpd': 'mov', 'vmovlps': 'mov', 'vmovmskpd': 'mov', 'vmovntdq': 'mov', 'vmovntps': 'mov', 'vmovq': 'mov', 'vmovsd': 'mov', 'vmovshdup': 'mov', 'vmovsldup': 'mov', 'vmovss': 'mov', 'vmovupd': 'mov', 'vmovups': 'mov', 'vmsave': 'save', 'vmulpd': 'mul', 'vmulps': 'mul', 'vmulsd': 'mul', 'vmulss': 'mul', 'vorpd': 'or', 'vorps': 'or', 'vpaddb': 'add', 'vpaddd': 'add', 'vpaddq': 'add', 'vpaddsb': 'add', 'vpaddsw': 'add', 'vpaddusb': 'add', 'vpaddusw': 'add', 'vpaddw': 'add', 'vpand': 'and', 'vpandd': 'and', 'vpandn': 'and', 'vpandq': 'and', 'vpclmulqdq': 'mul', 'vpcmpeqb': 'cmp', 'vpcmpeqd': 'cmp', 'vpcmpeqq': 'cmp', 'vpcmpeqw': 'cmp', 'vpcmpgtb': 'cmp', 'vpcmpgtd': 'cmp', 'vpcmpgtq': 'cmp', 'vpcmpgtw': 'cmp', 'vpcmpltw': 'cmp', 'vpcmpnleub': 'cmp', 'vphaddd': 'add', 'vphaddsw': 'add', 'vphaddw': 'add', 'vphminposuw': 'min', 'vphsubw': 'sub', 'vpmadd52huq': 'add', 'vpmadd52luq': 'add', 'vpmaddubsw': 'add', 'vpmaddwd': 'add', 'vpmaskmovd': 'mov', 'vpmaskmovq': 'mov', 'vpmaxsb': 'max', 'vpmaxsd': 'max', 'vpmaxsw': 'max', 'vpmaxub': 'max', 'vpmaxud': 'max', 'vpmaxuw': 'max', 'vpminsb': 'min', 'vpminsd': 'min', 'vpminsq': 'min', 'vpminsw': 'min', 'vpminub': 'min', 'vpminuq': 'min', 'vpminuw': 'min', 'vpmovdw': 'mov', 'vpmovmskb': 'mov', 'vpmovsxbw': 'mov', 'vpmovsxdq': 'mov', 'vpmovsxwd': 'mov', 'vpmovzxbd': 'mov', 'vpmovzxbq': 'mov', 'vpmovzxbw': 'mov', 'vpmovzxwd': 'mov', 'vpmovzxwq': 'mov', 'vpmuldq': 'mul', 'vpmulhrsw': 'mul', 'vpmulhuw': 'mul', 'vpmulhw': 'mul', 'vpmulld': 'mul', 'vpmullw': 'mul', 'vpmuludq': 'mul', 'vpor': 'or', 'vporq': 'or', 'vpshaw': 'sha', 'vpsubb': 'sub', 'vpsubd': 'sub', 'vpsubq': 'sub', 'vpsubsb': 'sub', 'vpsubsw': 'sub', 'vpsubusb': 'sub', 'vpsubusw': 'sub', 'vpsubw': 'sub', 'vptest': 'test', 'vpxor': 'xor', 'vpxord': 'xor', 'vpxorq': 'xor', 'vsubpd': 'sub', 'vsubps': 'sub', 'vsubsd': 'sub', 'vsubss': 'sub', 'vxorpd': 'xor', 'vxorps': 'xor', 'xabort': 'or', 'xaddb': 'add', 'xaddl': 'add', 'xaddq': 'add', 'xor': 'xor', 'xorb': 'xor', 'xorl': 'xor', 'xorpd': 'xor', 'xorps': 'xor', 'xorq': 'xor', 'xorw': 'xor', 'xrelease': 'lea', 'xrstor': 'or', 'xrstors': 'or', 'xsave': 'save', 'xsave64': 'save', 'xsavec': 'save', 'xsaveopt': 'save', 'xsaveopt64': 'save', 'xsaves': 'save', 'xsha1': 'sha', 'xsha256': 'sha', 'xstorerng': 'or', 'xtest': 'test'},
    'mull_cluster': {'fimull': 'mul', 'fmull': 'mul', 'imull': 'mul', 'mull': 'mul', 'pmulld': 'mul', 'pmullw': 'mul',
                     'smull': 'mul', 'umull': 'mul', 'vpmulld': 'mul', 'vpmullw': 'mul'}
}


OP_CODES = ['aaa', 'aad', 'aam', 'aas', 'adcb', 'adcl', 'adcq', 'adcw', 'adcxq', 'add', 'addb', 'addl', 'addpd',
            'addps', 'addq', 'adds', 'addsd', 'addss', 'addsubps', 'addw', 'adoxl', 'adoxq', 'adr', 'adrp', 'aesdec', 'aesdeclast', 'aesenc', 'aesenclast', 'aesimc', 'aeskeygenassist', 'and', 'andb', 'andl', 'andnl', 'andnpd', 'andnps', 'andnq', 'andpd', 'andps', 'andq', 'ands', 'andw', 'arpl', 'asr', 'b', 'bfi', 'bfxil', 'bics', 'bl', 'blcil', 'blendps', 'blr', 'blsmskq', 'bndldx', 'bndstx', 'bound', 'br', 'brk', 'bsfl', 'bsfq', 'bsfw', 'bsrl', 'bsrq', 'bswapl', 'bswapq', 'btcl', 'btcq', 'btl', 'btq', 'btrl', 'btrq', 'btsl', 'btsq', 'btsw', 'btw', 'bzhiq', 'calll', 'callq', 'callw', 'cbnz', 'cbtw', 'cbz', 'ccmp', 'cinc', 'clac', 'clc', 'cld', 'cldemote', 'clflush', 'cli', 'cltd', 'cltq', 'clts', 'clz', 'clzero', 'cmc', 'cmeq', 'cmn', 'cmovael', 'cmovaeq', 'cmovaew', 'cmoval', 'cmovaq', 'cmovaw', 'cmovbel', 'cmovbeq', 'cmovbew', 'cmovbl', 'cmovbq', 'cmovbw', 'cmovel', 'cmoveq', 'cmovew', 'cmovgel', 'cmovgeq', 'cmovgew', 'cmovgl', 'cmovgq', 'cmovgw', 'cmovlel', 'cmovleq', 'cmovlew', 'cmovll', 'cmovlq', 'cmovlw', 'cmovnel', 'cmovneq', 'cmovnew', 'cmovnol', 'cmovnoq', 'cmovnpl', 'cmovnpq', 'cmovnpw', 'cmovnsl', 'cmovnsq', 'cmovnsw', 'cmovol', 'cmovoq', 'cmovpl', 'cmovpq', 'cmovsl', 'cmovsq', 'cmovsw', 'cmp', 'cmpb', 'cmpeqpd', 'cmpeqps', 'cmpeqsd', 'cmpeqss', 'cmpl', 'cmplepd', 'cmpleps', 'cmplesd', 'cmpless', 'cmpltpd', 'cmpltps', 'cmpltsd', 'cmpltss', 'cmpneqpd', 'cmpneqps', 'cmpneqsd', 'cmpneqss', 'cmpnlepd', 'cmpnleps', 'cmpnlesd', 'cmpnless', 'cmpnltpd', 'cmpnltps', 'cmpnltsd', 'cmpordpd', 'cmpordps', 'cmpordsd', 'cmpordss', 'cmppd', 'cmpps', 'cmpq', 'cmpsb', 'cmpsl', 'cmpsq', 'cmpss', 'cmpsw', 'cmpunordpd', 'cmpunordps', 'cmpunordsd', 'cmpunordss', 'cmpw', 'cmpxchg16b', 'cmpxchg8b', 'cmpxchgb', 'cmpxchgl', 'cmpxchgq', 'cmpxchgw', 'cneg', 'comisd', 'comiss', 'cpuid', 'cqto', 'crc32b', 'crc32l', 'crc32q', 'crc32w', 'cs', 'csdb', 'csel', 'cset', 'csetm', 'csinc', 'csinv', 'cvtdq2pd', 'cvtdq2ps', 'cvtpd2dq', 'cvtpd2ps', 'cvtpi2ps', 'cvtps2dq', 'cvtps2pd', 'cvtps2pi', 'cvtsd2si', 'cvtsd2ss', 'cvtsi2sd', 'cvtsi2sdl', 'cvtsi2sdq', 'cvtsi2ss', 'cvtsi2ssl', 'cvtsi2ssq', 'cvtss2sd', 'cvtss2si', 'cvttpd2dq', 'cvttps2dq', 'cvttps2pi', 'cvttsd2si', 'cvttss2si', 'cwtd', 'cwtl', 'daa', 'das', 'data16', 'decb', 'decl', 'decq', 'decw', 'divb', 'divl', 'divpd', 'divps', 'divq', 'divsd', 'divss', 'divw', 'dmb', 'ds', 'dup', 'emms', 'encls', 'enclu', 'enclv', 'enter', 'eor', 'es', 'extr', 'f2xm1', 'fabs', 'fadd', 'faddl', 'faddp', 'fadds', 'fbld', 'fbstp', 'fchs', 'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb', 'fcmovnbe', 'fcmovne', 'fcmovnu', 'fcmovu', 'fcom', 'fcomi', 'fcoml', 'fcomp', 'fcompi', 'fcompl', 'fcompp', 'fcomps', 'fcoms', 'fcos', 'fcvtzs', 'fdecstp', 'fdiv', 'fdivl', 'fdivp', 'fdivr', 'fdivrl', 'fdivrp', 'fdivrs', 'fdivs', 'femms', 'ffree', 'ffreep', 'fiaddl', 'fiadds', 'ficoml', 'ficompl', 'ficomps', 'ficoms', 'fidivl', 'fidivrl', 'fidivrs', 'fidivs', 'fildl', 'fildll', 'filds', 'fimull', 'fimuls', 'fincstp', 'fistl', 'fistpl', 'fistpll', 'fistps', 'fists', 'fisttpl', 'fisttpll', 'fisttps', 'fisubl', 'fisubrl', 'fisubrs', 'fisubs', 'fld', 'fld1', 'fldcw', 'fldenv', 'fldl', 'fldl2e', 'fldl2t', 'fldlg2', 'fldln2', 'fldpi', 'flds', 'fldt', 'fldz', 'fmadd', 'fmov', 'fmul', 'fmull', 'fmulp', 'fmuls', 'fnclex', 'fninit', 'fnmsub', 'fnop', 'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'fpatan', 'fprem', 'fprem1', 'fptan', 'frintp', 'frndint', 'frstor', 'fs', 'fscale', 'fsin', 'fsincos', 'fsqrt', 'fst', 'fstl', 'fstp', 'fstpl', 'fstps', 'fstpt', 'fsts', 'fsub', 'fsubl', 'fsubp', 'fsubr', 'fsubrl', 'fsubrp', 'fsubrs', 'fsubs', 'ftst', 'fucom', 'fucomi', 'fucomp', 'fucompi', 'fucompp', 'fxam', 'fxch', 'fxrstor', 'fxsave', 'fxtract', 'fyl2x', 'fyl2xp1', 'getsec', 'gs', 'hint', 'hlt', 'hsubps', 'idivb', 'idivl', 'idivq', 'idivw', 'imulb', 'imull', 'imulq', 'imulw', 'inb', 'incb', 'incl', 'incq', 'incsspq', 'incw', 'inl', 'insb', 'insl', 'insw', 'int', 'int3', 'into', 'invd', 'invlpg', 'inw', 'iretl', 'iretq', 'iretw', 'ja', 'jae', 'jb', 'jbe', 'jcxz', 'je', 'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jmpq', 'jmpw', 'jne', 'jno', 'jnp', 'jns', 'jo', 'jp', 'jrcxz', 'js', 'kandb', 'kmovb', 'kmovd', 'kmovw', 'kshiftlb', 'kshiftrd', 'kshiftrw', 'kunpckbw', 'kunpckdq', 'kxnorb', 'kxnorw', 'lahf', 'larl', 'larq', 'lcalll', 'lcallq', 'lcallw', 'ld1', 'ld2', 'ld4', 'ldar', 'ldaxr', 'lddqu', 'ldmxcsr', 'ldnp', 'ldp', 'ldr', 'ldrb', 'ldrh', 'ldrsb', 'ldrsh', 'ldrsw', 'ldsl', 'ldsw', 'ldur', 'ldurb', 'ldurh', 'ldursb', 'ldursh', 'ldursw', 'ldxr', 'leal', 'leaq', 'leave', 'leaw', 'lesl', 'lesw', 'lfence', 'lfsl', 'lfsq', 'lgdtl', 'lgdtq', 'lgsl', 'lgsq', 'lidtl', 'lidtq', 'ljmpl', 'ljmpq', 'ljmpw', 'lldtw', 'lmsww', 'lock', 'lodsb', 'lodsl', 'lodsq', 'lodsw', 'loop', 'loope', 'loopne', 'lretl', 'lretq', 'lretw', 'lsl', 'lsll', 'lslq', 'lslw', 'lsr', 'lssl', 'lssq', 'ltrw', 'lwpins', 'lzcntl', 'madd', 'maskmovdqu', 'maskmovq', 'maxpd', 'maxps', 'maxsd', 'maxss', 'mfence', 'minpd', 'minps', 'minsd', 'minss', 'monitor', 'monitorx', 'mov', 'movabsb', 'movabsl', 'movabsq', 'movabsw', 'movapd', 'movaps', 'movb', 'movbel', 'movbeq', 'movd', 'movddup', 'movdiri', 'movdq2q', 'movdqa', 'movdqu', 'movhlps', 'movhpd', 'movhps', 'movi', 'movk', 'movl', 'movlhps', 'movlpd', 'movlps', 'movmskpd', 'movmskps', 'movntdq', 'movntdqa', 'movntil', 'movntiq', 'movntps', 'movntq', 'movq', 'movq2dq', 'movsb', 'movsbl', 'movsbq', 'movsbw', 'movsd', 'movshdup', 'movsl', 'movsldup', 'movslq', 'movsq', 'movss', 'movsw', 'movswl', 'movswq', 'movupd', 'movups', 'movw', 'movzbl', 'movzbq', 'movzbw', 'movzwl', 'movzwq', 'mpsadbw', 'mrs', 'msr', 'msub', 'mul', 'mulb', 'mull', 'mulpd', 'mulps', 'mulq', 'mulsd', 'mulss', 'mulw', 'mulxq', 'mvn', 'mwait', 'mwaitx', 'neg', 'negb', 'negl', 'negq', 'negw', 'nop', 'nopl', 'nopq', 'nopw', 'notb', 'notl', 'notq', 'notw', 'orb', 'orl', 'orn', 'orpd', 'orps', 'orq', 'orr', 'orw', 'outb', 'outl', 'outsb', 'outsl', 'outsw', 'outw', 'pabsb', 'pabsd', 'pabsw', 'packssdw', 'packsswb', 'packusdw', 'packuswb', 'paddb', 'paddd', 'paddq', 'paddsb', 'paddsw', 'paddusb', 'paddusw', 'paddw', 'palignr', 'pand', 'pandn', 'pause', 'pavgb', 'pavgusb', 'pavgw', 'pblendvb', 'pblendw', 'pclmulqdq', 'pcmpeqb', 'pcmpeqd', 'pcmpeqq', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw', 'pcmpistri', 'pconfig', 'pextrb', 'pextrd', 'pextrq', 'pextrw', 'pf2id', 'pf2iw', 'pfacc', 'pfadd', 'pfcmpge', 'pfcmpgt', 'pfmax', 'pfmin', 'pfmul', 'pfnacc', 'pfrcpit2', 'pfrsqit1', 'pfsub', 'pfsubr', 'phaddd', 'phaddsw', 'phaddw', 'phminposuw', 'phsubd', 'phsubw', 'pi2fd', 'pi2fw', 'pinsrb', 'pinsrd', 'pinsrq', 'pinsrw', 'pmaddubsw', 'pmaddwd', 'pmaxsb', 'pmaxsd', 'pmaxsw', 'pmaxub', 'pmaxud', 'pmaxuw', 'pminsb', 'pminsd', 'pminsw', 'pminub', 'pminud', 'pminuw', 'pmovmskb', 'pmovsxbd', 'pmovsxbq', 'pmovsxbw', 'pmovsxdq', 'pmovsxwd', 'pmovsxwq', 'pmovzxbd', 'pmovzxbq', 'pmovzxbw', 'pmovzxdq', 'pmovzxwd', 'pmovzxwq', 'pmuldq', 'pmulhrsw', 'pmulhrw', 'pmulhuw', 'pmulhw', 'pmulld', 'pmullw', 'pmuludq', 'popal', 'popaw', 'popcntl', 'popfl', 'popfq', 'popfw', 'popl', 'popq', 'popw', 'por', 'prefetch', 'prefetchnta', 'prefetcht0', 'prefetcht1', 'prefetcht2', 'prefetchw', 'prefetchwt1', 'prfm', 'prfum', 'psadbw', 'pshufb', 'pshufd', 'pshufhw', 'pshuflw', 'pshufw', 'psignb', 'psignd', 'psignw', 'pslld', 'pslldq', 'psllq', 'psllw', 'psrad', 'psraw', 'psrld', 'psrldq', 'psrlq', 'psrlw', 'psubb', 'psubd', 'psubq', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'psubw', 'pswapd', 'ptest', 'punpckhbw', 'punpckhdq', 'punpckhqdq', 'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklqdq', 'punpcklwd', 'pushal', 'pushaw', 'pushfl', 'pushfq', 'pushfw', 'pushl', 'pushq', 'pushw', 'pxor', 'rclb', 'rcll', 'rclq', 'rclw', 'rcpps', 'rcpss', 'rcrb', 'rcrl', 'rcrq', 'rcrw', 'rdmsr', 'rdpkru', 'rdpmc', 'rdrandl', 'rdrandq', 'rdseedl', 'rdseedq', 'rdsspq', 'rdtsc', 'rdtscp', 'rep', 'repne', 'ret', 'retl', 'retq', 'retw', 'rev', 'rev16', 'rev32', 'rev64', 'rex64', 'rolb', 'roll', 'rolq', 'rolw', 'ror', 'rorb', 'rorl', 'rorq', 'rorw', 'rorxl', 'rorxq', 'rsm', 'rsqrtps', 'rsqrtss', 'sahf', 'salc', 'sarb', 'sarl', 'sarq', 'sarw', 'sarxl', 'sarxq', 'sbbb', 'sbbl', 'sbbq', 'sbbw', 'sbfiz', 'scasb', 'scasl', 'scasq', 'scasw', 'scvtf', 'sdiv', 'serialize', 'seta', 'setae', 'setb', 'setbe', 'sete', 'setg', 'setge', 'setl', 'setle', 'setne', 'setno', 'setnp', 'setns', 'seto', 'setp', 'sets', 'sfence', 'sgdtl', 'sgdtq', 'sgdtw', 'sha1msg1', 'sha1msg2', 'sha1nexte', 'sha1rnds4', 'sha256msg1', 'sha256msg2', 'sha256rnds2', 'shlb', 'shldl', 'shldq', 'shll', 'shlq', 'shlw', 'shlxl', 'shlxq', 'shrb', 'shrdl', 'shrdq', 'shrl', 'shrq', 'shrw', 'shrxl', 'shrxq', 'shufpd', 'shufps', 'sidtl', 'sidtq', 'skinit', 'sldtl', 'sldtq', 'sldtw', 'smswl', 'smswq', 'smsww', 'smull', 'sqrtpd', 'sqrtps', 'sqrtsd', 'sqrtss', 'ss', 'st1', 'st2', 'st3', 'st4', 'stac', 'stc', 'std', 'stgi', 'sti', 'stlr', 'stlxr', 'stmxcsr', 'stosb', 'stosl', 'stosq', 'stosw', 'stp', 'str', 'strb', 'strh', 'strl', 'strq', 'strw', 'stur', 'sturb', 'sturh', 'stxr', 'stxrb', 'sub', 'subb', 'subl', 'subpd', 'subps', 'subq', 'subs', 'subsd', 'subss', 'subw', 'svc', 'swapgs', 'sxtb', 'sxth', 'sxtw', 'syscall', 'sysenter', 'sysexitl', 'sysexitq', 'sysretl', 'sysretq', 'tbnz', 'tbz', 'testb', 'testl', 'testq', 'testw', 'tst', 'tzcntl', 'tzcntq', 'uaddlv', 'ubfiz', 'ubfx', 'ucomisd', 'ucomiss', 'ud1l', 'ud1w', 'ud2', 'udf', 'udiv', 'umaxv', 'uminv', 'umull', 'unpckhpd', 'unpckhps', 'unpcklpd', 'unpcklps', 'uxtb', 'uxth', 'vaddpd', 'vaddps', 'vaddsd', 'vaddss', 'vaddsubpd', 'vaddsubps', 'vaesdec', 'vaesdeclast', 'vaesenc', 'vaesenclast', 'valignd', 'valignq', 'vandnpd', 'vandnps', 'vandpd', 'vandps', 'vblendmps', 'vblendpd', 'vblendvps', 'vbroadcastf128', 'vbroadcasti128', 'vbroadcasti32x4', 'vbroadcasti32x8', 'vbroadcastss', 'vcmpeq_uqsd', 'vcmpeq_uqss', 'vcmpeqpd', 'vcmpeqps', 'vcmpeqsd', 'vcmpeqss', 'vcmpfalse_ossd', 'vcmpge_oqsd', 'vcmpgt_oqsd', 'vcmpgtps', 'vcmpgtss', 'vcmplepd', 'vcmpleps', 'vcmplesd', 'vcmplt_oqsd', 'vcmpneq_ussd', 'vcmpnge_uqpd', 'vcmpnge_uqss', 'vcmpngt_uqps', 'vcmpngtps', 'vcmpnltpd', 'vcmpnltsd', 'vcmpordps', 'vcmppd', 'vcmpps', 'vcmpsd', 'vcmpss', 'vcomisd', 'vcomiss', 'vcvtdq2pd', 'vcvtdq2ps', 'vcvtpd2dq', 'vcvtpd2psy', 'vcvtph2ps', 'vcvtps2pd', 'vcvtps2ph', 'vcvtqq2pd', 'vcvtsd2si', 'vcvtsd2ss', 'vcvtsi2sdl', 'vcvtsi2ss', 'vcvtsi2ssl', 'vcvtss2sd', 'vcvtss2si', 'vcvttpd2dq', 'vcvttpd2dqy', 'vcvttpd2qq', 'vcvttps2dq', 'vcvttps2qq', 'vcvttsd2si', 'vdivpd', 'vdivps', 'vdivsd', 'vdivss', 'vdppd', 'vdpps', 'verr', 'verw', 'vextractf128', 'vextracti128', 'vextracti32x4', 'vextracti32x8', 'vextracti64x4', 'vextractps', 'vfixupimmpd', 'vfixupimmps', 'vfmadd132ps', 'vfmadd132sd', 'vfmadd132ss', 'vfmadd213ps', 'vfmadd213sd', 'vfmadd213ss', 'vfmadd231sd', 'vfmadd231ss', 'vfmaddsub132ps', 'vfmaddsubpd', 'vfmaddsubps', 'vfmsub132sd', 'vfmsub213sd', 'vfmsub213ss', 'vfmsub231ps', 'vfmsubadd132ps', 'vfmsubadd231pd', 'vfmsubadd231ps', 'vfmsubaddpd', 'vfmsubpd', 'vfmsubps', 'vfmsubss', 'vfnmadd132ps', 'vfnmadd132sd', 'vfnmadd213ps', 'vfnmadd213sd', 'vfnmadd231sd', 'vfnmaddps', 'vfnmaddss', 'vfnmsub132ps', 'vfnmsub231sd', 'vfnmsubpd', 'vfnmsubps', 'vfnmsubsd', 'vgf2p8affineqb', 'vhaddpd', 'vhaddps', 'vhsubpd', 'vhsubps', 'vinsertf128', 'vinserti128', 'vinserti32x4', 'vinserti32x8', 'vinsertps', 'vlddqu', 'vldmxcsr', 'vmaskmovpd', 'vmaskmovps', 'vmaxpd', 'vmaxps', 'vmaxsd', 'vmaxss', 'vmcall', 'vmfunc', 'vminpd', 'vminps', 'vminsd', 'vminss', 'vmlaunch', 'vmload', 'vmmcall', 'vmovapd', 'vmovaps', 'vmovd', 'vmovddup', 'vmovdqa', 'vmovdqa32', 'vmovdqa64', 'vmovdqu', 'vmovdqu32', 'vmovdqu64', 'vmovdqu8', 'vmovhpd', 'vmovhps', 'vmovlpd', 'vmovlps', 'vmovmskpd', 'vmovntdq', 'vmovntps', 'vmovq', 'vmovsd', 'vmovshdup', 'vmovsldup', 'vmovss', 'vmovupd', 'vmovups', 'vmptrld', 'vmptrst', 'vmreadl', 'vmreadq', 'vmresume', 'vmrun', 'vmsave', 'vmulpd', 'vmulps', 'vmulsd', 'vmulss', 'vmwritel', 'vmwriteq', 'vmxoff', 'vorpd', 'vorps', 'vpabsb', 'vpabsd', 'vpabsw', 'vpackssdw', 'vpacksswb', 'vpackusdw', 'vpackuswb', 'vpaddb', 'vpaddd', 'vpaddq', 'vpaddsb', 'vpaddsw', 'vpaddusb', 'vpaddusw', 'vpaddw', 'vpalignr', 'vpand', 'vpandd', 'vpandn', 'vpandq', 'vpavgb', 'vpavgw', 'vpblendd', 'vpblendmb', 'vpblendmd', 'vpblendmq', 'vpblendmw', 'vpblendvb', 'vpblendw', 'vpbroadcastb', 'vpbroadcastd', 'vpbroadcastq', 'vpbroadcastw', 'vpclmulqdq', 'vpcmpeqb', 'vpcmpeqd', 'vpcmpeqq', 'vpcmpeqw', 'vpcmpgtb', 'vpcmpgtd', 'vpcmpgtq', 'vpcmpgtw', 'vpcmpltw', 'vpcmpnleub', 'vpcomq', 'vpcomub', 'vpcomud', 'vpdpbusd', 'vpdpwssd', 'vpdpwssds', 'vperm2f128', 'vperm2i128', 'vpermb', 'vpermd', 'vpermi2b', 'vpermi2pd', 'vpermilps', 'vpermq', 'vpermt2b', 'vpermt2q', 'vpextrb', 'vpextrd', 'vpextrq', 'vpextrw', 'vpgatherdd', 'vpgatherdq', 'vphaddd', 'vphaddsw', 'vphaddw', 'vphminposuw', 'vphsubw', 'vpinsrb', 'vpinsrd', 'vpinsrq', 'vpinsrw', 'vpmacsdqh', 'vpmacssdd', 'vpmacssdql', 'vpmacssww', 'vpmadcswd', 'vpmadd52huq', 'vpmadd52luq', 'vpmaddubsw', 'vpmaddwd', 'vpmaskmovd', 'vpmaskmovq', 'vpmaxsb', 'vpmaxsd', 'vpmaxsw', 'vpmaxub', 'vpmaxud', 'vpmaxuw', 'vpminsb', 'vpminsd', 'vpminsq', 'vpminsw', 'vpminub', 'vpminuq', 'vpminuw', 'vpmovdw', 'vpmovmskb', 'vpmovsxbw', 'vpmovsxdq', 'vpmovsxwd', 'vpmovzxbd', 'vpmovzxbq', 'vpmovzxbw', 'vpmovzxwd', 'vpmovzxwq', 'vpmuldq', 'vpmulhrsw', 'vpmulhuw', 'vpmulhw', 'vpmulld', 'vpmullw', 'vpmuludq', 'vpor', 'vporq', 'vprold', 'vprotb', 'vprotd', 'vprotq', 'vprotw', 'vpsadbw', 'vpscatterdd', 'vpshaw', 'vpshld', 'vpshldd', 'vpshldw', 'vpshlw', 'vpshrdd', 'vpshrdvd', 'vpshrdw', 'vpshufb', 'vpshufbitqmb', 'vpshufd', 'vpshufhw', 'vpshuflw', 'vpsignb', 'vpsignd', 'vpsignw', 'vpslld', 'vpslldq', 'vpsllq', 'vpsllvd', 'vpsllvq', 'vpsllvw', 'vpsllw', 'vpsrad', 'vpsravd', 'vpsraw', 'vpsrld', 'vpsrldq', 'vpsrlq', 'vpsrlvd', 'vpsrlvq', 'vpsrlw', 'vpsubb', 'vpsubd', 'vpsubq', 'vpsubsb', 'vpsubsw', 'vpsubusb', 'vpsubusw', 'vpsubw', 'vpternlogd', 'vptest', 'vpunpckhbw', 'vpunpckhdq', 'vpunpckhqdq', 'vpunpckhwd', 'vpunpcklbw', 'vpunpckldq', 'vpunpcklqdq', 'vpunpcklwd', 'vpxor', 'vpxord', 'vpxorq', 'vrangeps', 'vrcp14ss', 'vrcpps', 'vrcpss', 'vrndscaless', 'vroundsd', 'vroundss', 'vrsqrtps', 'vrsqrtss', 'vshufi32x4', 'vshufi64x2', 'vshufpd', 'vshufps', 'vsqrtpd', 'vsqrtps', 'vsqrtsd', 'vsqrtss', 'vstmxcsr', 'vsubpd', 'vsubps', 'vsubsd', 'vsubss', 'vucomisd', 'vucomiss', 'vunpckhpd', 'vunpckhps', 'vunpcklpd', 'vunpcklps', 'vxorpd', 'vxorps', 'vzeroall', 'vzeroupper', 'wait', 'wbinvd', 'wrmsr', 'wrpkru', 'wrssd', 'xabort', 'xacquire', 'xaddb', 'xaddl', 'xaddq', 'xbegin', 'xchgb', 'xchgl', 'xchgq', 'xchgw', 'xcryptcbc', 'xcryptcfb', 'xcryptctr', 'xcryptecb', 'xcryptofb', 'xend', 'xgetbv', 'xlatb', 'xorb', 'xorl', 'xorpd', 'xorps', 'xorq', 'xorw', 'xrelease', 'xrstor', 'xrstors', 'xsave', 'xsave64', 'xsavec', 'xsaveopt', 'xsaveopt64', 'xsaves', 'xsha1', 'xsha256', 'xstorerng', 'xtest', 'yield']


REVERSED_MAP = {'lretl': 0, 'retl': 0, 'iretl': 0, 'lretw': 0, 'lretq': 0, 'sbbb': 1, 'subl': 1, 'sbbl': 1, 'subb': 1,
                'sbbq': 1, 'fsubl': 1, 'fsubs': 1, 'fsubp': 1, 'fsub': 1, 'fsubr': 1, 'fisubrs': 1, 'subw': 1, 'fisubrl': 1, 'fisubl': 1, 'sbbw': 1, 'fsubrs': 1, 'psubsb': 1, 'fisubs': 1, 'fsubrl': 1, 'psubb': 1, 'bsfl': 1, 'ss': 1, 'vsubsd': 1, 'subsd': 1, 'subss': 1, 'fsubrp': 1, 'psubw': 1, 'psubsw': 1, 'psubusb': 1, 'subps': 1, 'psubusw': 1, 'sub': 1, 'b': 1, 'subs': 1, 'pfsub': 1, 'pshufb': 1, 'pfsubr': 1, 'subls': 1, 'vfmsubps': 1, 'vpsubusw': 1, 'vsubss': 1, 'vsubps': 1, 'movl': 2, 'movsb': 2, 'movb': 2, 'movw': 2, 'movsl': 2, 'movaps': 2, 'movsd': 2, 'movsbq': 2, 'cmovneq': 2, 'movq': 2, 'movswl': 2, 'movabsq': 2, 'movslq': 2, 'cmovll': 2, 'movswq': 2, 'movsbl': 2, 'movsbw': 2, 'movsw': 2, 'movdqa': 2, 'cmovnel': 2, 'movhps': 2, 'movlps': 2, 'movdqu': 2, 'cmovsq': 2, 'movd': 2, 'cmovsl': 2, 'cmovoq': 2, 'cmovsw': 2, 'movabsb': 2, 'movlhps': 2, 'movapd': 2, 'movss': 2, 'cmovol': 2, 'cmovnol': 2, 'cmovnsl': 2, 'vmovsd': 2, 'vmovq': 2, 'vmovdqa': 2, 'cmovnsq': 2, 'cmovnpl': 2, 'movabsl': 2, 'vmovdqu': 2, 'movlt': 2, 'movhs': 2, 'moveq': 2, 'movs': 2, 'movlo': 2, 'movls': 2, 'vmov': 2, 'movle': 2, 'vmovaps': 2, 'vmovups': 2, 'movsq': 2, 'movhlps': 2, 'vmovapd': 2, 'cmovnoq': 2, 'movabsw': 2, 'vmovd': 2, 'vmovlps': 2, 'daa': 3, 'aaa': 3, 'aas': 3, 'aad': 3, 'aam': 3, 'lcalll': 4, 'shll': 4, 'calll': 4, 'shrl': 4, 'callq': 4, 'shrdl': 4, 'lesl': 4, 'lodsl': 4, 'ldsl': 4, 'lsll': 4, 'shldl': 4, 'psllq': 4, 'sldtl': 4, 'pslld': 4, 'callw': 4, 'lsls': 4, 'ldrhlo': 4, 'lsrs': 4, 'lsr': 4, 'lsrls': 4, 'ldrsh': 4, 'lsl': 4, 'lslls': 4, 'lslvs': 4, 'lsrpl': 4, 'lslhs': 4, 'lslhi': 4, 'ldrhvs': 4, 'lsrlo': 4, 'ldrh': 4, 'lfsl': 4, 'lcallq': 4, 'lssl': 4, 'lcallw': 4, 'lgsl': 4, 'lslq': 4, 'ldrhls': 4, 'lsrhi': 4, 'fdivr': 5, 'fdivl': 5, 'divl': 5, 'idivl': 5, 'fidivrs': 5, 'fdivrs': 5, 'fdivrp': 5, 'fidivl': 5, 'fidivrl': 5, 'fdiv': 5, 'fidivs': 5, 'fdivs': 5, 'fdivrl': 5, 'fdivp': 5, 'sete': 6, 'setne': 6, 'setge': 6, 'setle': 6, 'setbe': 6, 'setae': 6, 'imull': 7, 'mull': 7, 'fmul': 7, 'fmuls': 7, 'fmulp': 7, 'fimull': 7, 'fmull': 7, 'mul': 7, 'umlal': 7, 'smull': 7, 'umull': 7, 'pfmul': 7, 'cmoveq': 8, 'cmovel': 8, 'cmovaeq': 8, 'cmovbel': 8, 'cmovlq': 8, 'cmovlel': 8, 'cmovgel': 8, 'cmovael': 8, 'cmovgeq': 8, 'cmovbeq': 8, 'cmovleq': 8, 'prefetchnta': 9, 'prefetchw': 9, 'prefetch': 9, 'prefetcht0': 9, 'prefetchwt1': 9, 'prefetcht2': 9, 'prefetcht1': 9, 'addl': 10, 'addb': 10, 'das': 10, 'addq': 10, 'addw': 10, 'fadds': 10, 'faddp': 10, 'fadd': 10, 'fiadds': 10, 'faddl': 10, 'paddw': 10, 'addps': 10, 'addpd': 10, 'subpd': 10, 'addsd': 10, 'andpd': 10, 'paddd': 10, 'vaddsd': 10, 'pand': 10, 'vpand': 10, 'paddusb': 10, 'addss': 10, 'paddusw': 10, 'vhsubpd': 10, 'paddsb': 10, 'andps': 10, 'psubd': 10, 'ds': 10, 'pandn': 10, 'paddsw': 10, 'vpaddd': 10, 'pmaddwd': 10, 'paddb': 10, 'andnps': 10, 'vpaddusb': 10, 'phaddsw': 10, 'paddq': 10, 'andnpd': 10, 'adds': 10, 'and': 10, 'add': 10, 'ands': 10, 'vaddss': 10, 'vaddpd': 10, 'pfadd': 10, 'vpaddq': 10, 'vandpd': 10, 'vhaddps': 10, 'vaddsubpd': 10, 'vpaddw': 10, 'pmaddubsw': 10, 'vandnpd': 10, 'vandnps': 10, 'vpsubd': 10, 'vpaddusw': 10, 'vaddps': 10, 'pswapd': 10, 'vfmaddps': 10, 'vsubpd': 10, 'haddps': 10, 'vandps': 10, 'ficoml': 11, 'fucomp': 11, 'fcomp': 11, 'fcom': 11, 'fcompl': 11, 'fucomi': 11, 'fucom': 11, 'fcomps': 11, 'fcoml': 11, 'fcoms': 11, 'ficoms': 11, 'fcmovu': 11, 'ficomps': 11, 'ficompl': 11, 'fucompp': 11, 'fcompp': 11, 'fcompi': 11, 'fcos': 11, 'fucompi': 11, 'fcomi': 11, 'fstps': 12, 'fstp': 12, 'fstpl': 12, 'fstpt': 12, 'fistl': 12, 'fsts': 12, 'ftst': 12, 'fisttpll': 12, 'fstl': 12, 'fistps': 12, 'fistpl': 12, 'fisttps': 12, 'fistpll': 12, 'fisttpl': 12, 'fists': 12, 'fbstp': 12, 'cmpl': 13, 'cmpsl': 13, 'pcmpeqw': 13, 'pcmpeqb': 13, 'cmpeqsd': 13, 'pcmpeqd': 13, 'cmpsq': 13, 'cmpltpd': 13, 'cmpps': 13, 'cmpeqps': 13, 'cmpltss': 13, 'cmpnltps': 13, 'cmpltps': 13, 'pfcmpge': 13, 'cmpneqps': 13, 'cmpleps': 13, 'pfcmpeq': 13, 'vpcmpeqw': 13, 'cmpss': 13, 'cmpless': 13, 'cmpnltsd': 13, 'cmplesd': 13, 'cmpltsd': 13, 'stmpl': 13, 'vcmpps': 13, 'pushl': 14, 'pushfl': 14, 'pushq': 14, 'pushw': 14, 'shufps': 14, 'pshuflw': 14, 'pshufhw': 14, 'pushfw': 14, 'pshufw': 14, 'push': 14, 'vpush': 14, 'pushaw': 14, 'vpshufhw': 14, 'vpshuflw': 14, 'punpckhwd': 15, 'unpcklps': 15, 'unpckhpd': 15, 'punpcklwd': 15, 'unpcklpd': 15, 'punpckldq': 15, 'punpckhbw': 15, 'punpcklbw': 15, 'punpckhdq': 15, 'unpckhps': 15, 'punpcklqdq': 15, 'punpckhqdq': 15, 'vpunpcklqdq': 15, 'vpunpcklwd': 15, 'vunpckhpd': 15, 'vpunpckhwd': 15, 'vpunpckhdq': 15, 'fldln2': 16, 'fldt': 16, 'fldl2t': 16, 'fldl2e': 16, 'fldlg2': 16, 'cvttps2pi': 17, 'cvtdq2ps': 17, 'cvtps2pd': 17, 'cvtdq2pd': 17, 'vcvtdq2pd': 17, 'cvtpd2ps': 17, 'cvtps2dq': 17, 'cvtps2pi': 17, 'cvtpi2ps': 17, 'cvttpd2dq': 17, 'cvttps2dq': 17, 'cvtpi2pd': 17, 'vcvtdq2ps': 17, 'vcvtpd2dq': 17, 'cvtpd2dq': 17, 'cmovew': 18, 'cmovnew': 18, 'cmovaew': 18, 'cmovgew': 18, 'cmovbew': 18, 'pcmpgtb': 19, 'pcmpgtd': 19, 'pcmpgtw': 19, 'pfcmpgt': 19, 'vpcmpgtb': 19, 'vfmadd213sd': 20, 'vfmadd231sd': 20, 'vfnmadd132sd': 20, 'vfnmadd231sd': 20, 'vfmadd213ss': 20, 'cvtsi2sd': 21, 'cvtss2sd': 21, 'cvtsi2ss': 21, 'cvttsd2si': 21, 'cvtsd2ss': 21, 'cvttss2si': 21, 'cvtsi2sdl': 21, 'cvtsd2si': 21, 'cvtsi2ssl': 21, 'cvtsi2sdq': 21, 'cvtsi2ssq': 21, 'cvtss2si': 21, 'vcvtsd2ss': 21, 'vcvtsi2ssl': 21, 'vcvtss2sd': 21, 'sqrtps': 22, 'rsqrtps': 22, 'sqrtsd': 22, 'str': 22, 'strd': 22, 'vrsqrtps': 22, 'pfrsqrt': 22, 'rsqrtss': 22, 'sqrtss': 22, 'pxor': 23, 'vpor': 23, 'vxorps': 23, 'vpxor': 23, 'vxorpd': 23, 'cmpxchgl': 24, 'cmpxchgq': 24, 'cmpxchgb': 24, 'cmpxchg8b': 24, 'cmpxchg16b': 24, 'itett': 25, 'ittee': 25, 'itte': 25, 'iteet': 25, 'itt': 25, 'ite': 25, 'iteee': 25, 'it': 25, 'itete': 25, 'ittte': 25, 'ittt': 25, 'itee': 25, 'ittet': 25, 'itet': 25, 'mcr2': 26, 'mcr': 26, 'mrrc2': 26, 'mrc': 26, 'mrc2': 26, 'mcrr': 26, 'mcrr2': 26, 'vldr': 27, 'ldrvs': 27, 'vpsrld': 27, 'vpsrlvd': 27, 'mulsd': 28, 'mulps': 28, 'mulss': 28, 'muls': 28, 'vmulss': 28, 'rorb': 29, 'rorl': 29, 'rorw': 29, 'rorq': 29, 'rors': 29, 'orr': 29, 'orrs': 29, 'popl': 30, 'popq': 30, 'popw': 30, 'vpop': 30, 'pop': 30, 'pshufd': 31, 'shufpd': 31, 'vshufps': 31, 'vshufpd': 31, 'vpshufb': 31, 'vpshufd': 31, 'ucomiss': 32, 'vcomisd': 32, 'comiss': 32, 'vcomiss': 32}


def get_op_code_dictionary(op_codes):
    return {x: [] for x in op_codes}
    # return {x: [] for x in list(set(REVERSED_MAP.values()))}


def get_mean(arr):
    if len(arr) == 0:
        return 0
    return sum(arr) / len(arr)


def get_min(arr):
    if len(arr) == 0:
        return 0
    return min(arr)


def get_max(arr):
    if len(arr) == 0:
        return 0
    return max(arr)


def get_median(arr):
    if len(arr) == 0:
        return 0
    arr = sorted(arr)
    if len(arr) % 2 == 0:
        return (arr[(len(arr)-1) // 2] + arr[len(arr) // 2]) / 2
    return arr[(len(arr)-1) // 2]


def reduce_op_code_list_dictionary_to_jump_image(
        d,
        total_lines,
        full_file_path,
        op_codes
):
    if total_lines == 0:
        print(full_file_path)
        return 0
    im = np.zeros((len(op_codes), len(op_codes), 3))
    for ix, x in enumerate(op_codes):
        # for iy, y in enumerate(OP_CODES):
        differences = []
        for i in d[x]:
            for j in d[x]:
                differences += [abs(i - j)]

        print(x, len(differences))
        # im[ix, iy, 0] = 255 * (get_min(differences) / total_lines)
        # im[ix, iy, 1] = 255 * (get_mean(differences) / total_lines)
        # im[ix, iy, 2] = 255 * (get_max(differences) / total_lines)

    new_image = Image.fromarray(im.astype('uint8'), mode='HSV')
    new_image.convert('RGB').save(full_file_path, 'PNG')


def get_leading_digit(x):
    if (x // 10) == 0:
        return x
    return get_leading_digit(x // 10)


def reduce_op_code_list_dictionary_to_one_jump_leading_digit_bins(d, total_lines):
    if total_lines == 0:
        return 0

    bins = {x: 0 for x in range(1, 10)}
    for ix, x in enumerate(list(set(REVERSED_MAP.values()))):
        number_of_operation_instances = len(d[x])
        # print(number_of_operation_instances)
        if number_of_operation_instances > 1:
            for jump_index in range(number_of_operation_instances - 1):
                key = get_leading_digit(d[x][jump_index+1] - d[x][jump_index])
                bins[key] += 1

    total = sum(bins.values())
    for x in bins:
        bins[x] = bins[x] / total

    return bins


def reduce_op_code_list_to_counts(op_code_occurence_list, file_operations=None):

    temp = {}
    for line_index, line in enumerate(op_code_occurence_list):
        operation = line
        if len(operation.split()) > 1:
            operation = operation.split()[0]

        if operation not in temp:
            temp.update({operation: 0})
        temp[operation] += 1

    if file_operations is None:
        file_operations = {}
    else:
        for x in list(set(list(file_operations.keys()) + list(temp.keys()))):
            if x in temp and temp[x] >= 2 and x in file_operations:
                file_operations[x] += 1
            elif x in temp and temp[x] >= 2 and x not in file_operations:
                file_operations.update({x: 1})


def reduce_op_code_list_to_index_list(
        op_code_occurence_list,
        op_codes,
        # cluster_map=None
):
    not_clustered = True #cluster_map is None
    # keys = op_codes if not_clustered else list(set(cluster_map.values()))
    file_operations = get_op_code_dictionary(op_codes=op_codes)
    # print(keys)
    line_index = 0

    for line_index, line in enumerate(op_code_occurence_list):
        operation = line
        if len(operation.split()) > 1:
            operation = operation.split()[0]
        # print(operation)
        # if not_clustered:
        if operation in file_operations:
            file_operations[operation] += [line_index]
        # else:
        #     if operation in op_codes:
        #         file_operations[cluster_map[operation]] += [line_index]
            # file_operations[REVERSED_MAP[operation]] += [line_index]

    return file_operations, len(op_code_occurence_list)


def reduce_multiple_op_code_lists_to_index_lists(
        op_code_occurence_list,
        op_code_options_dictionary
):

    file_operations = {
        key: get_op_code_dictionary(op_codes=value) for key, value in op_code_options_dictionary.items()
    }

    for line_index, line in enumerate(op_code_occurence_list):
        operation = line
        if len(operation.split()) > 1:
            operation = operation.split()[0]

        for k in file_operations:
            if operation in file_operations[k]:
                file_operations[k][operation] += [line_index]

    for op_code_option in file_operations:
        if op_code_option in OP_CODE_CLUSTER:
            temp = {x: [] for x in set(OP_CODE_CLUSTER[op_code_option].values())}
            for op in file_operations[op_code_option]:
                temp[OP_CODE_CLUSTER[op_code_option][op]] += file_operations[op_code_option][op]

            for op in temp:
                temp[op] = sorted(temp[op])
            file_operations[op_code_option] = temp

    return file_operations, len(op_code_occurence_list)


def difference_by_dictionary(x, y):

    if list(x.keys()).sort() != list(y.keys()).sort():
        print(f'Keys in first dictionary - {x.keys}')
        print(f'Keys in second dictionary - {y.keys}')
        raise Exception('Dictionaries do not contain matching keys.')

    for x_key in x:
        x[x_key] = abs(x[x_key] - y[x_key])

    diff_sum = sum(x.values())

    return diff_sum


def cluster_by_op():
    focus = ['add', 'and', 'call', 'cmp', 'div', 'jmp', 'lea', 'loop', 'max', 'min', 'mov', 'mul', 'or', 'pop', 'push',
             'ret', 'set', 'shll', 'save', 'sha', 'sub', 'sys', 'test', 'xor']
    ops = {x: [] for x in focus}

    extras = []
    for op in OP_CODES:
        added = False
        for i in focus:
            if i in op: # and len(op) <= len(i) + 2:
                ops[i] += [op]
                added = True
        if not added:
            extras += [op]

    arr = []
    flip = {x: x for x in focus}
    print(ops)
    for op, o in ops.items():
        arr += o
        for k in o:
            # print(k)
            flip.update({k: op})
    print(sorted(set(arr + focus)))
    print({f: flip[f] for f in sorted(flip.keys())})
    # for f in focus:
    #     print(f)
    #     print(ops[f])
    # print('---')
    # print(extras)


if __name__ == "__main__":
    # cluster_by_op()
    pass


