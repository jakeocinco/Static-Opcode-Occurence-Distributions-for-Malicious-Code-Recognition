import os
import random
import numpy as np
import scipy.stats as ss
import math
import matplotlib.pyplot as plt
import matplotlib.image as im

from sklearn.cluster import DBSCAN

from op_codes import *

# Read 1k files of each

# random.shuffle(arr)


# # op_codes = ['aaa', 'aad', 'aam', 'aas', 'adc', 'adcb', 'adcl', 'adcq', 'adcs', 'adcw', 'add', 'addb', 'addeq', 'addl', 'addmi', 'addpd', 'addps', 'addq', 'adds', 'addsd', 'addss', 'addvc', 'addw', 'adr', 'adrls', 'adrmi', 'aesdec', 'aesdeclast', 'and', 'andb', 'andl', 'andnl', 'andnpd', 'andnps', 'andpd', 'andps', 'andq', 'ands', 'andw', 'arpl', 'asreq', 'asrle', 'asrmi', 'asrs', 'asrvc', 'asrvs', 'b', 'beq', 'bfc', 'bfi', 'bge', 'bgt', 'bhi', 'bhs', 'bic', 'bics', 'bicvs', 'bkpt', 'bl', 'blcil', 'ble', 'blendvps', 'blo', 'bls', 'blt', 'blx', 'bmi', 'bndcl', 'bndldx', 'bndstx', 'bne', 'bound', 'bpl', 'bsfl', 'bsfw', 'bsrl', 'bsrw', 'bswapl', 'bswapq', 'bswapw', 'btcl', 'btcw', 'btl', 'btq', 'btrl', 'btrq', 'btrw', 'btsl', 'btsq', 'btsw', 'btw', 'bvc', 'bvs', 'bx', 'calll', 'callq', 'callw', 'cbnz', 'cbtw', 'cbz', 'cdp', 'cdp2', 'clc', 'cld', 'cldemote', 'clflush', 'cli', 'cltd', 'cltq', 'clts', 'clz', 'cmc', 'cmn', 'cmovael', 'cmovaeq', 'cmovaew', 'cmoval', 'cmovaq', 'cmovaw', 'cmovbel', 'cmovbeq', 'cmovbl', 'cmovbq', 'cmovbw', 'cmovel', 'cmoveq', 'cmovew', 'cmovgel', 'cmovgeq', 'cmovgew', 'cmovgl', 'cmovgq', 'cmovgw', 'cmovlel', 'cmovleq', 'cmovll', 'cmovlq', 'cmovlw', 'cmovnel', 'cmovneq', 'cmovnew', 'cmovnol', 'cmovnpl', 'cmovnpq', 'cmovnsl', 'cmovnsq', 'cmovol', 'cmovoq', 'cmovpl', 'cmovsl', 'cmovsq', 'cmovsw', 'cmp', 'cmpb', 'cmpeqps', 'cmpeqsd', 'cmpge', 'cmpl', 'cmplesd', 'cmpltpd', 'cmpltps', 'cmpltsd', 'cmpmi', 'cmpneqpd', 'cmpneqps', 'cmpnlepd', 'cmpnltsd', 'cmppd', 'cmppl', 'cmpps', 'cmpq', 'cmpsb', 'cmpsd', 'cmpsl', 'cmpsq', 'cmpsw', 'cmpunordps', 'cmpw', 'cmpxchg8b', 'cmpxchgb', 'cmpxchgl', 'cmpxchgq', 'comisd', 'comiss', 'cpuid', 'cqto', 'crc32b', 'crc32l', 'cs', 'cvtdq2pd', 'cvtdq2ps', 'cvtpd2dq', 'cvtpd2ps', 'cvtpi2ps', 'cvtps2dq', 'cvtps2pd', 'cvtps2pi', 'cvtsd2si', 'cvtsd2ss', 'cvtsi2sd', 'cvtsi2sdl', 'cvtsi2sdq', 'cvtsi2ss', 'cvtsi2ssl', 'cvtsi2ssq', 'cvtss2sd', 'cvtss2si', 'cvttpd2dq', 'cvttps2pi', 'cvttsd2si', 'cvttss2si', 'cwtd', 'cwtl', 'daa', 'das', 'data16', 'decb', 'decl', 'decq', 'decw', 'divb', 'divl', 'divps', 'divq', 'divsd', 'divss', 'divw', 'dmb', 'ds', 'emms', 'enter', 'eor', 'eors', 'es', 'f2xm1', 'fabs', 'fadd', 'faddl', 'faddp', 'fadds', 'fbld', 'fbstp', 'fchs', 'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb', 'fcmovnbe', 'fcmovne', 'fcmovnu', 'fcmovu', 'fcom', 'fcomi', 'fcoml', 'fcomp', 'fcompi', 'fcompl', 'fcompp', 'fcomps', 'fcoms', 'fcos', 'fdecstp', 'fdiv', 'fdivl', 'fdivp', 'fdivr', 'fdivrl', 'fdivrp', 'fdivrs', 'fdivs', 'femms', 'ffree', 'ffreep', 'fiaddl', 'fiadds', 'ficoml', 'ficompl', 'ficomps', 'ficoms', 'fidivl', 'fidivrl', 'fidivrs', 'fidivs', 'fildl', 'fildll', 'filds', 'fimull', 'fimuls', 'fincstp', 'fistl', 'fistpl', 'fistpll', 'fistps', 'fists', 'fisttpl', 'fisttpll', 'fisttps', 'fisubl', 'fisubrl', 'fisubrs', 'fisubs', 'fld', 'fld1', 'fldcw', 'fldenv', 'fldl', 'fldl2e', 'fldl2t', 'fldlg2', 'fldln2', 'fldpi', 'flds', 'fldt', 'fldz', 'fmul', 'fmull', 'fmulp', 'fmuls', 'fnclex', 'fninit', 'fnop', 'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'fpatan', 'fprem', 'fprem1', 'fptan', 'frndint', 'frstor', 'fs', 'fscale', 'fsin', 'fsincos', 'fsqrt', 'fst', 'fstl', 'fstp', 'fstpl', 'fstps', 'fstpt', 'fsts', 'fsub', 'fsubl', 'fsubp', 'fsubr', 'fsubrl', 'fsubrp', 'fsubrs', 'fsubs', 'ftst', 'fucom', 'fucomi', 'fucomp', 'fucompi', 'fucompp', 'fxam', 'fxch', 'fxrstor', 'fxsave', 'fxtract', 'fyl2x', 'fyl2xp1', 'getsec', 'gs', 'hint', 'hintgt', 'hlt', 'hsubpd', 'idivb', 'idivl', 'idivq', 'idivw', 'imulb', 'imull', 'imulq', 'imulw', 'inb', 'incb', 'incl', 'incq', 'incw', 'inl', 'insb', 'insl', 'insw', 'int', 'int3', 'into', 'invd', 'invlpg', 'inw', 'iretl', 'iretq', 'iretw', 'it', 'ite', 'itee', 'iteee', 'iteet', 'itet', 'itete', 'itett', 'itt', 'itte', 'ittee', 'ittet', 'ittt', 'ittte', 'itttt', 'ja', 'jae', 'jb', 'jbe', 'jcxz', 'je', 'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpl', 'jmpq', 'jmpw', 'jne', 'jno', 'jnp', 'jns', 'jo', 'jp', 'jrcxz', 'js', 'kaddb', 'kandnb', 'kandnw', 'kandw', 'kmovb', 'kunpckbw', 'kxnorb', 'kxnorw', 'kxorb', 'lahf', 'larl', 'lcalll', 'lcallq', 'lcallw', 'ldc', 'ldc2', 'ldc2l', 'ldcl', 'lddqu', 'ldm', 'ldmdb', 'ldmxcsr', 'ldr', 'ldrb', 'ldrbls', 'ldrbvc', 'ldrd', 'ldrex', 'ldrexb', 'ldrh', 'ldrheq', 'ldrhlo', 'ldrhlt', 'ldrhmi', 'ldrhvs', 'ldrls', 'ldrmi', 'ldrsb', 'ldrsh', 'ldrvc', 'ldrvs', 'ldsl', 'ldsw', 'leal', 'leaq', 'leave', 'leaw', 'lesl', 'lesw', 'lfence', 'lfsl', 'lgdtl', 'lgdtq', 'lgsl', 'lidtl', 'lidtq', 'ljmpl', 'ljmpq', 'ljmpw', 'lldtw', 'lmsww', 'lock', 'lodsb', 'lodsl', 'lodsq', 'lodsw', 'loop', 'loope', 'loopne', 'lretl', 'lretq', 'lretw', 'lsl', 'lslge', 'lslhi', 'lslhs', 'lsll', 'lslls', 'lsllt', 'lslmi', 'lslne', 'lslpl', 'lslq', 'lsls', 'lslvc', 'lslvs', 'lslw', 'lsr', 'lsreq', 'lsrhs', 'lsrlo', 'lsrls', 'lsrmi', 'lsrpl', 'lsrs', 'lssl', 'ltrw', 'maskmovq', 'maxps', 'maxsd', 'maxss', 'mcr', 'mcr2', 'mcrr', 'mcrr2', 'mfence', 'minps', 'minsd', 'minss', 'mla', 'mov', 'movabsb', 'movabsl', 'movabsq', 'movabsw', 'movapd', 'movaps', 'movb', 'movd', 'movddup', 'movdqa', 'movdqu', 'moveq', 'movge', 'movgt', 'movhi', 'movhlps', 'movhps', 'movhs', 'movl', 'movle', 'movlhps', 'movlo', 'movlpd', 'movlps', 'movls', 'movlt', 'movmi', 'movmskps', 'movne', 'movntdq', 'movntil', 'movntiq', 'movntps', 'movntq', 'movntsd', 'movq', 'movs', 'movsb', 'movsbl', 'movsbq', 'movsbw', 'movsd', 'movshdup', 'movsl', 'movslq', 'movsq', 'movss', 'movsw', 'movswl', 'movswq', 'movsww', 'movt', 'movupd', 'movups', 'movvs', 'movw', 'movzbl', 'movzbq', 'movzbw', 'movzwl', 'movzwq', 'movzww', 'mrc', 'mrc2', 'mrrc2', 'mul', 'mulb', 'mull', 'mulpd', 'mulps', 'mulq', 'muls', 'mulsd', 'mulss', 'mulw', 'mvn', 'mvns', 'mwaitx', 'negb', 'negl', 'negq', 'negw', 'nop', 'nopl', 'nopw', 'notb', 'notl', 'notq', 'notw', 'orb', 'orl', 'orns', 'orpd', 'orps', 'orq', 'orr', 'orrhs', 'orrs', 'orw', 'outb', 'outl', 'outsb', 'outsl', 'outsw', 'outw', 'pabsb', 'pabsd', 'packssdw', 'packsswb', 'packuswb', 'paddb', 'paddd', 'paddq', 'paddsb', 'paddsw', 'paddusb', 'paddusw', 'paddw', 'palignr', 'pand', 'pandn', 'pause', 'pavgb', 'pavgw', 'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw', 'pcmpistri', 'pconfig', 'pextrw', 'pf2id', 'pfadd', 'pfcmpeq', 'pfcmpge', 'pfmax', 'pfmin', 'pfmul', 'pfnacc', 'pfsub', 'phaddsw', 'pi2fd', 'pi2fw', 'pinsrb', 'pinsrw', 'pkhbt', 'pmaddubsw', 'pmaddwd', 'pmaxsw', 'pmaxub', 'pminsw', 'pminub', 'pmovmskb', 'pmovzxbd', 'pmulhuw', 'pmulhw', 'pmullw', 'pmuludq', 'pop', 'popal', 'popaw', 'popfl', 'popfq', 'popfw', 'pophi', 'pophs', 'popl', 'popmi', 'popq', 'popw', 'por', 'prefetch', 'prefetchnta', 'prefetcht0', 'prefetcht1', 'prefetcht2', 'prefetchw', 'prefetchwt1', 'psadbw', 'pshufb', 'pshufd', 'pshufhw', 'pshuflw', 'pshufw', 'psignb', 'psignw', 'pslld', 'pslldq', 'psllq', 'psllw', 'psrad', 'psraw', 'psrld', 'psrldq', 'psrlq', 'psrlw', 'psubb', 'psubd', 'psubq', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'psubw', 'punpckhbw', 'punpckhdq', 'punpckhqdq', 'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklqdq', 'punpcklwd', 'push', 'pushal', 'pushaw', 'pushfl', 'pushfq', 'pushfw', 'pushl', 'pushq', 'pushvc', 'pushw', 'pxor', 'rclb', 'rcll', 'rclq', 'rclw', 'rcpps', 'rcrb', 'rcrl', 'rcrq', 'rcrw', 'rdmsr', 'rdpmc', 'rdrandl', 'rdseedl', 'rdtsc', 'rep', 'repne', 'retl', 'retq', 'retw', 'rev', 'rev16', 'revsh', 'rex64', 'rolb', 'roll', 'rolq', 'rolw', 'rorb', 'rorl', 'rorq', 'rors', 'rorw', 'rsb', 'rsbhs', 'rsbmi', 'rsbs', 'rsm', 'rsqrtps', 'rstorssp', 'sahf', 'salc', 'sarb', 'sarl', 'sarq', 'sarw', 'sbbb', 'sbbl', 'sbbq', 'sbbw', 'sbc', 'sbcs', 'sbfx', 'scasb', 'scasl', 'scasq', 'scasw', 'seta', 'setae', 'setb', 'setbe', 'sete', 'setend', 'setg', 'setge', 'setl', 'setle', 'setne', 'setno', 'setnp', 'setns', 'seto', 'setp', 'sets', 'sev', 'sfence', 'sgdtl', 'sgdtq', 'sha1rnds4', 'sha256msg1', 'shlb', 'shldl', 'shldw', 'shll', 'shlq', 'shlw', 'shrb', 'shrdl', 'shrdw', 'shrl', 'shrq', 'shrw', 'shufpd', 'shufps', 'sidtl', 'sidtq', 'sldtl', 'sldtw', 'smswl', 'smsww', 'smull', 'sqrtps', 'sqrtsd', 'sqrtss', 'ss', 'stc', 'stc2', 'stc2l', 'stcl', 'std', 'stgi', 'sti', 'stm', 'stmdb', 'stmmi', 'stmxcsr', 'stosb', 'stosl', 'stosq', 'stosw', 'str', 'strb', 'strbge', 'strd', 'strex', 'strexb', 'strh', 'strhne', 'strhvc', 'strl', 'strmi', 'strvs', 'strw', 'sub', 'subb', 'subl', 'subls', 'submi', 'subpd', 'subpl', 'subps', 'subq', 'subs', 'subsd', 'subss', 'subvs', 'subw', 'svc', 'sxtab', 'sxtb', 'sxth', 'syscall', 'sysenter', 'sysexitl', 'sysretl', 't1mskcl', 'tbb', 'tbh', 'testb', 'testl', 'testq', 'testw', 'trap', 'tst', 'ubfx', 'ucomisd', 'ucomiss', 'ud1l', 'ud2', 'udf', 'umlal', 'umull', 'unpckhpd', 'unpckhps', 'unpcklpd', 'unpcklps', 'uxtb', 'uxth', 'vaddpd', 'vaddps', 'vaddsd', 'vaddss', 'vaddsubpd', 'vaddsubps', 'vaesdec', 'vandnpd', 'vandnps', 'vandpd', 'vandps', 'vblendpd', 'vblendvps', 'vcmpeqpd', 'vcmpeqps', 'vcmplesd', 'vcmpless', 'vcmpngepd', 'vcmpnltsd', 'vcmppd', 'vcmpps', 'vcmpsd', 'vcmpss', 'vcomisd', 'vcomiss', 'vcvtdq2pd', 'vcvtpd2dq', 'vcvtpd2dqy', 'vcvtqq2pd', 'vcvtsd2ss', 'vcvtsi2sdl', 'vcvtsi2ss', 'vcvtsi2ssl', 'vcvtss2sd', 'vcvttpd2dq', 'vcvttpd2dqx', 'vcvttsd2si', 'vdivpd', 'vdivps', 'vdivsd', 'vdivss', 'verr', 'verw', 'vfixupimmps', 'vfmadd132pd', 'vfmadd132ss', 'vfmadd213pd', 'vfmadd213sd', 'vfmadd231sd', 'vfmadd231ss', 'vfmaddps', 'vfmaddsub213ps', 'vfmaddsubpd', 'vfmsub132sd', 'vfmsub213sd', 'vfmsubadd132pd', 'vfmsubadd213pd', 'vfmsubps', 'vfnmadd132ss', 'vfnmadd213ps', 'vfnmsubps', 'vhaddpd', 'vhaddps', 'vhsubpd', 'vhsubps', 'vldr', 'vmaxpd', 'vmaxps', 'vmaxsd', 'vmaxss', 'vminpd', 'vminps', 'vminsd', 'vminss', 'vmload', 'vmmcall', 'vmov', 'vmovapd', 'vmovaps', 'vmovd', 'vmovddup', 'vmovdqa', 'vmovdqu', 'vmovhpd', 'vmovhps', 'vmovlhps', 'vmovlps', 'vmovntps', 'vmovq', 'vmovsd', 'vmovshdup', 'vmovss', 'vmovupd', 'vmovups', 'vmptrld', 'vmptrst', 'vmreadl', 'vmreadq', 'vmrs', 'vmrun', 'vmsr', 'vmulpd', 'vmulps', 'vmulsd', 'vmulss', 'vmwritel', 'vmwriteq', 'vmxoff', 'vorpd', 'vorps', 'vp4dpwssds', 'vpackssdw', 'vpacksswb', 'vpackuswb', 'vpaddb', 'vpaddd', 'vpaddq', 'vpaddsb', 'vpaddsw', 'vpaddusb', 'vpaddusw', 'vpaddw', 'vpand', 'vpandn', 'vpavgb', 'vpavgw', 'vpblendmb', 'vpblendmd', 'vpcmpeqb', 'vpcmpeqd', 'vpcmpeqw', 'vpcmpgtb', 'vpcmpgtd', 'vpcmpgtw', 'vpcomuw', 'vpcomw', 'vpermi2b', 'vpermil2ps', 'vpermilps', 'vpermq', 'vphaddd', 'vphaddsw', 'vpinsrw', 'vpmacsdql', 'vpmaddwd', 'vpmaskmovd', 'vpmaxsw', 'vpmaxub', 'vpmaxuq', 'vpminsb', 'vpminsq', 'vpminsw', 'vpminub', 'vpmovmskb', 'vpmulhw', 'vpmulld', 'vpmullw', 'vpmuludq', 'vpop', 'vpor', 'vprotb', 'vpsadbw', 'vpshaq', 'vpshlb', 'vpshldd', 'vpshlq', 'vpshufb', 'vpshufbitqmb', 'vpslld', 'vpslldq', 'vpsllq', 'vpsllw', 'vpsrad', 'vpsravd', 'vpsraw', 'vpsrld', 'vpsrlq', 'vpsrlw', 'vpsubb', 'vpsubd', 'vpsubq', 'vpsubsb', 'vpsubsw', 'vpsubusb', 'vpsubusw', 'vpsubw', 'vpunpckhbw', 'vpunpckhdq', 'vpunpckhqdq', 'vpunpcklbw', 'vpunpckldq', 'vpunpcklqdq', 'vpunpcklwd', 'vpush', 'vpxor', 'vrcpps', 'vrcpss', 'vroundsd', 'vrsqrt28sd', 'vrsqrtss', 'vshuff64x2', 'vshufpd', 'vshufps', 'vsqrtpd', 'vsqrtps', 'vsqrtsd', 'vsqrtss', 'vstr', 'vsubpd', 'vsubps', 'vsubsd', 'vsubss', 'vucomisd', 'vucomiss', 'vunpckhpd', 'vunpckhps', 'vunpcklpd', 'vunpcklps', 'vxorpd', 'vxorps', 'vzeroupper', 'wait', 'wbinvd', 'wrmsr', 'xabort', 'xacquire', 'xaddb', 'xaddl', 'xaddq', 'xaddw', 'xbegin', 'xchgb', 'xchgl', 'xchgq', 'xchgw', 'xcryptcfb', 'xcryptecb', 'xcryptofb', 'xgetbv', 'xlatb', 'xorb', 'xorl', 'xorpd', 'xorps', 'xorq', 'xorw', 'xrelease', 'xrstor', 'xrstors', 'xsave', 'xsavec', 'xsaveopt', 'xsaves', 'xtest']
# op_codes = ['xorl', 'je', 'jne', 'movl', 'addl', 'cmpl', 'jb', 'andl', 'movb', 'jae', 'subl', 'orl', 'jbe', 'incl', 'testl', 'cmpb', 'ja', 'jmp', 'sbbl', 'decl', 'addb', 'testb', 'leal', 'movw', 'xorb', 'jle', 'imull', 'js', 'andb', 'jge', 'int3', 'jns', 'jl', 'orb', 'nop', 'subb', 'adcl', 'jg', 'rep', 'xchgl', 'cmpw', 'pushl', 'popl', 'sbbb', 'negl', 'shrl', 'retl', 'cltd', 'adcb', 'shll', 'sarl', 'calll', 'movzwl', 'lock', 'sete', 'leave', 'notl', 'cld', 'stosl', 'movzbl', 'jmpl', 'movsl', 'testw', 'incb', 'repne', 'inb', 'jp', 'jo', 'outsb', 'outsl', 'std', 'insb', 'divl', 'pushfl', 'mull', 'insl', 'setne', 'jnp', 'shlb', 'wait', 'rolb', 'cwtl', 'stosb', 'loopne', 'lretl', 'movsb', 'outb', 'popal', 'inl', 'roll', 'movsbl', 'outl', 'sahf', 'int', 'jno', 'bound', 'fldl', 'arpl', 'xlatb', 'xchgb', 'addw', 'enter', 'clc', 'lodsb', 'idivl', 'scasb', 'rorl', 'loop', 'faddl', 'sarb', 'decb', 'popfl', 'das', 'ljmpl', 'lahf', 'lcalll', 'pushal', 'cmpsb', 'rcrl', 'lodsl', 'cmpsl', 'iretl', 'aaa', 'stc', 'hlt', 'daa', 'loope', 'flds', 'cli', 'andw', 'scasl', 'aas', 'fildl', 'fmull', 'rclb', 'btl', 'shrb', 'aam', 'cmc', 'lesl', 'rorb', 'fstpl', 'orw', 'movswl', 'negb', 'imulw', 'fstps', 'into', 'fnstsw', 'sti', 'fadds', 'salc', 'incw', 'jecxz', 'rcrb', 'fiaddl', 'ldsl', 'fcompl', 'movq', 'fnstcw', 'aad', 'outsw', 'subw', 'cmovel', 'filds', 'fiadds', 'fldcw', 'fmuls', 'btsl', 'fstp', 'xaddl', 'rcll', 'fdivl', 'fldt', 'setg', 'fcomps', 'fsubs', 'cmovnel', 'fstl', 'notb', 'fstpt', 'fdivr', 'sldtw', 'pushw', 'fld', 'fsubrs', 'fcomp', 'fadd', 'movups', 'movdqa', 'fmulp', 'setl', 'fsub', 'fsubl', 'fildll', 'decw', 'cpuid', 'fsubrl', 'faddp', 'fnclex', 'fimull', 'fldz', 'fdivs', 'setge', 'fcom', 'fcoml', 'xorw', 'fcoms', 'movd', 'fistpll', 'fidivl', 'fxch', 'setb', 'shrdl', 'fdivrp', 'fdivrs', 'fmul', 'ficoml', 'fdivrl', 'popaw', 'fimuls', 'seta', 'shldl', 'fsubr', 'frstor', 'fistpl', 'divb', 'fbstp', 'fldenv', 'fisttpll', 'cmoval', 'fdivp', 'movaps', 'fsts', 'imulb', 'fnstenv', 'fdiv', 'movdqu', 'fnsave', 'fisttps', 'ficoms', 'ficomps', 'fisttpl', 'cmovbl', 'idivb', 'fsubp', 'mulb', 'fchs', 'movsw', 'fld1']
# op_codes = sorted(op_codes[:150])


def get_top_op_codes(
        bins=100,
        num_files=1000,
        top=50,
        verbose=False
):
    arr = os.listdir("/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    clean = list(
        filter(
            lambda x: 'clean' in x,
            arr
        )
    )[:num_files]
    infected = list(
        filter(
            lambda x: 'infect' in x,
            arr
        )
    )[:num_files]

    files = clean + infected

    x = {}
    b = random.randint(10, 30)

    for i, file_name in enumerate(files):
        with open(
                f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/{file_name}"
        ) as file:
            if verbose:
                if i % 100 == 99:
                    print(i + 1)
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

            file_x = {}
            for line_index, line in enumerate(file_data):
                operation = line
                if len(operation.split()) > 1:
                    operation = operation.split()[0]

                if line in op_codes:
                    if operation not in file_x:
                        file_x.update({operation: 0})
                    file_x[operation] += 1

            for op in file_x:
                if file_x[op] > 1 and op in op_codes:
                    if op not in x:
                        x.update({op: 0})
                    x[op] += 1

    l = list({k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}.keys())[:top]
    print(l)
    print([x[i] for i in l])


def get_independence_distance_matrix(
        distributions,
        op_code_type,
        bins=100,
        num_files=1000,
):

    op_codes = OP_CODE_DICT[op_code_type]

    probs = {x: np.zeros(num_files * 2) for x in op_codes}
    probs = {distribution: probs.copy() for distribution in distributions}

    ranks = {x: None for x in op_codes}
    ranks = {distribution: ranks.copy() for distribution in distributions}

    arr = os.listdir("/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    clean = list(
        filter(
            lambda x: 'clean' in x,
            arr
        )
    )[:num_files]
    infected = list(
        filter(
            lambda x: 'infect' in x,
            arr
        )
    )[:num_files]

    files = clean + infected

    x = []
    b = 5

    for i, file_name in enumerate(files):
        with open(
                f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/{file_name}"
        ) as file:

            # if i % 100 == 99:
            #     print(i + 1)
            try:
                file_data = str(file.read()).split()
            except:
                print(file_name)

            file_operations, line_index = reduce_op_code_list_to_index_list(
                file_data,
                op_codes=op_codes
            )

            for _, op in enumerate(op_codes):

                data = {distribution: np.zeros((bins, )) for distribution in distributions}
                for distribution in distributions:
                    data[distribution] += 1

                if op in file_operations:
                    number_of_operation_instances = len(file_operations[op])
                    if number_of_operation_instances > 1:
                        for jump_index in range(number_of_operation_instances - 1):
                            jump = file_operations[op][jump_index + 1] - file_operations[op][jump_index]

                            for distribution, map_func in distributions.items():
                                mapped_jump = map_func(jump, bins)
                                if mapped_jump < 1:
                                    key = int((mapped_jump * bins) // 1)
                                    data[distribution][key] += 1

                for distribution, map_func in distributions.items():
                    probs[distribution][op][i] = (sum(data[distribution][b:]) / sum(data[distribution])) # + (random.random() / 1000)

    for distribution, map_func in distributions.items():
        for op in op_codes:
            ranks[distribution][op] = ss.rankdata(probs[distribution][op])

    for distribution, map_func in distributions.items():
        independence_matrix = np.zeros((len(op_codes), len(op_codes)))

        for i, op1 in enumerate(op_codes):
            for k, op2 in enumerate(op_codes):
                if op1 == op2:
                    independence_matrix[i, k] = 1
                    # independence_matrix[i, k] = 0
                else:
                    N = num_files * 2
                    D = sum(
                        [(ranks[distribution][op1][j] - ranks[distribution][op2][j])**2 for j in range(N)]
                    )
                    denom = N * ((N**2) - 1)
                    rs = 1 - (D * 6 / denom)
                    Z = rs * math.sqrt(N - 1)

                    independence_matrix[i, k] = ss.norm.cdf(Z)
                    # independence_matrix[i, k] = 2 * min(ss.norm.cdf(Z), ss.norm.cdf(Z * -1))

        directory = f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset" \
                    f"/op_code_relation_samples/negative_correlation_matrices/{op_code_type}/"
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        # print(independence_matrix)
        with open(f"{directory}/independence.npy", 'wb') as file:
            np.save(file, independence_matrix)

        independence_matrix *= 255
        independence_matrix = independence_matrix.astype(int)
        plt.imsave(
            f"{directory}/independence.png",
            independence_matrix,
            cmap='gray'
        )

        # plt.imshow(independence_matrix, interpolation=None, cmap='gray')
        # plt.show()


def cluster_indepentents(
        distance_matrix,
        op_code_type,
        eps=0.1
):

    op_codes = OP_CODE_DICT[op_code_type]

    indexes = [i for i in range(distance_matrix.shape[0])]

    ind_map = {
        i: {
            i: r for i, r in enumerate(distance_matrix[i])
        } for i in range(distance_matrix.shape[0])
    }
    val_map = {
        i: {
            r + (random.random() / 10000): i for i, r in enumerate(distance_matrix[i])
        } for i in range(distance_matrix.shape[0])
    }
    random.shuffle(indexes)

    clusters = {}

    # print(distance_matrix)

    while len(indexes) > 0:
        if indexes[0] not in clusters:
            clusters.update({indexes[0]: []})
        lowest_val = min(val_map[indexes[0]])
        lowest_index = val_map[indexes[0]][lowest_val]

        # print(indexes[0], lowest_index, lowest_val)
        # print(clusters[indexes[0]])

        if lowest_val < eps:
            add_to_cluster = True
            # for i in clusters[indexes[0]]:
            #     if ind_map[lowest_index][i] > eps:
            #         add_to_cluster = False

            if add_to_cluster:
                clusters[indexes[0]] += [lowest_index]

        # print(lowest_index, lowest_val)
        # print(val_map[indexes[0]])
        # if indexes[0] != lowest_index:
        val_map[indexes[0]].pop(lowest_val)
        # ind_map[indexes[0]].pop(lowest_index)

        if not list(val_map[indexes[0]].keys()):
            indexes.remove(indexes[0])

    top = (-1, [])
    for c in clusters:

        l = list(
            map(
                lambda x: op_codes[x],
                [c] + clusters[c]
            )
        )
        if len(l) > top[0]:
            top = (len(l), list(l))

        if len(l) > 4:
            print(len(l), list(l))

    print(f"top({top[0]}) - {top[1]}")
    print()



#           add to prob list for op p(X > b) for op

# for op1
#       for op2
#           rank for every file which they both occur?? (or not)
#           calculate prob of independence and write to distance matrix

# Cluster based on 'distance matrix', with goal of clustering dependent distributions

distributions = {
    'linear': lambda a, b: a / 1000,
    # 'log10': lambda a, b: math.log(1 + ((a / 1000) * 9), 10),
    # 'log100': lambda a, b: math.log(1 + ((a / 1000) * 99), 100),
    # 'threshold': lambda a, b: a / b
}

# get_top_op_codes(
#     top=250,
#     # num_files=25
# )

for op in ['benign', 'infected', 'union', 'intersection', 'disjoint']:
    get_independence_distance_matrix(
        distributions=distributions,
        op_code_type=op,
        bins=25,
        num_files=500
    )

    # distance_matrix = np.load(
    #     f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/"
    #     f"independence_matrices/{op}/independence.npy"
    # )
    # with open(
    #         ) as file:
    #     try:
    #         file_data = file.read()
    #     except:
    #         print('file_name')

    print(f" -- {op} -- ")
    # distance_matrix = file_data[:, :, 0]
    # distance_matrix = distance_matrix.astype(float)
    # distance_matrix *= (1 / 255)
    # print(distance_matrix)
    # print()

    # cluster_indepentents(
    #     distance_matrix,
    #     op_code_type=op,
    #     eps=.5
    # )

