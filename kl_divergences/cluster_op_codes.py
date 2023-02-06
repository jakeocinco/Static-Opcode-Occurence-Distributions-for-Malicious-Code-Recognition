from op_codes import *
import math
from sklearn.cluster import DBSCAN
import numpy as np

def embedding(x):

    samples = 'abcdefghijklmnopqrstuvwxyz0123456789'
    embed = [0 for _ in range(len(samples))]

    for l in x:
        embed[samples.find(l)] += 1

    sum_sqrs = sum([x**2 for x in embed])
    embed = [(x / math.sqrt(sum_sqrs)) for x in embed]
    return embed

DATA = {
    -1: ['jbe', 'orl', 'cmc', 'insl', 'ja', 'incl', 'je', 'outsl', 'adcl', 'jle', 'outb', 'andl', 'leal', 'insb', 'jne',
         'jae', 'lahf', 'std', 'cmpb', 'jno', 'arpl', 'retw', 'loop', 'js', 'imulw', 'testl', 'jge', 'outsb', 'xchgl',
         'pushal', 'scasb', 'xorb', 'jb', 'jns', 'stc', 'orb', 'decl', 'bound', 'jnp', 'jo', 'leave', 'cli', 'popal',
         'xorl', 'salc', 'enter', 'jp', 'adcb', 'outl', 'andb', 'xchgb', 'cmpsb', 'jmp', 'jl', 'lock', 'nop', 'movzwl',
         'negl', 'cltd', 'inl', 'xaddl', 'notl', 'int3', 'cld', 'seto', 'repne', 'jg', 'rolb', 'rclb', 'loope', 'roll',
         'sarb', 'testb', 'rcrb', 'testw', 'popfl', 'cmpw', 'sarl', 'rep', 'stosl', 'jmpl', 'xlatb', 'inb', 'loopne',
         'movzbl', 'lodsb', 'andq', 'nopw', 'negw', 'imulq', 'btsl', 'divq', 'jmpq', 'movups', 'negq', 'movntiq',
         'shrq', 'bswapq', 'btl', 'notq', 'cmovgl', 'decq', 'cmoval', 'cmpq', 'andw', 'xorq', 'testq', 'shlq', 'subq',
         'cltq', 'setg', 'retq', 'cmovbq', 'nopl', 'sarq', 'btrl', 'incb', 'incq', 'leaq', 'xorw', 'orq', 'orw', 'fxch',
         'ljmpl', 'decb', 'shlb', 'stmxcsr', 'fchs', 'incw', 'movzbw', 'cpuid', 'fnstsw', 'fnstcw', 'rcrl', 'flds',
         'notb', 'ldmxcsr', 'setl', 'stosb', 'stosw', 'negb', 'wait', 'int', 'fldl', 'fldz', 'idivb', 'fcmovnb', 'divw',
         'rcrw', 'fcmove', 'sldtw', 'cwtl', 'sti', 'sarw', 'imulb', 'adcw', 'clc', 'filds', 'xaddb', 'fiaddl', 'scasl',
         'fxtract', 'frstor', 'fsqrt', 'fbld', 'ljmpw', 'sahf', 'fldenv', 'fcmovb', 'jecxz', 'xbegin', 'hlt', 'decw',
         'mulw', 'shlw', 'fildl', 'shrb', 'rcll', 'into', 'fscale', 'rolw', 'fimuls', 'ffreep', 'idivw', 'fcmovbe',
         'xrelease', 'divb', 'fld', 'fnstenv', 'rclw', 'xacquire', 'fnsave', 'fyl2xp1', 'xchgw', 'shrw', 'notw',
         'fsincos', 'fld1', 'mulb', 'fildll', 'fldcw', 'outsw', 'popaw', 'sysretl', 'pmuludq', 'sysenter', 'larl',
         'strw', 'pavgb', 'cldemote', 'clts', 'sgdtl', 'rsm', 'wbinvd', 'fcmovnu', 'ucomisd', 'pmovmskb', 'bsrl', 'btq',
         'movzbq', 'movzwq', 'por', 'cmovbl', 'xorps', 'pcmpistri', 'mulq', 'sets', 'idivq', 'psrldq', 'cmovgq',
         'comisd', 'seta', 'cmovaq', 'cqto', 'divsd', 'packsswb', 'maxps', 'syscall', 'setp', 'rdpmc', 'data16',
         'rex64', 'gs', 'cs', 'pushfq', 'popfq', 'fs', 'invd', 'jrcxz', 'palignr', 'f2xm1', 'pextrw', 'mulpd', 'fnclex',
         'movlpd', 'setb', 'fxam', 'xorpd', 'orpd', 'fabs', 'pause', 'frndint', 'fyl2x', 'setns', 'rcpps', 'psrlq',
         'cbtw', 'lgdtl', 'verw', 'invlpg', 'pmulhuw', 'vpmovmskb', 'btsq', 'xchgq', 'vpsrlq', 'btrq', 'movntps',
         'vmulsd', 'fcmovne', 'lidtl', 'bswapl', 'pavgw', 'fcmovnbe', 'fnop', 'lldtw', 'femms', 'fst', 'ltrw', 'divps',
         'verr', 'emms', 'ud2', 'divss', 'cmpnlepd', 'outw', 'rdtsc', 'insw', 'vpsravd', 'adcq', 'rolq', 'psraw',
         'pinsrw', 'es', 'pminsw', 'pmullw', 'crc32l', 'psrlw', 'movntq', 'psllw', 'packuswb', 'pslldq', 'psrld',
         'psadbw', 'pmulhw', 'psrad', 'pmaxsw', 'cmovlw', 'btcl', 'btw', 'sgdtq', 'scasq', 'rdseedl', 'iretq', 'vmulpd',
         'lodsq', 'wrmsr', 'stosq', 'packssdw', 'rcrq', 'psubq', 'rclq', 'fpatan', 'bndldx', 'fincstp', 'cmovaw',
         'iretw', 'vmreadl', 'setno', 'bndstx', 'ffree', 'lodsw', 'fsin', 'cmovpl', 'fldpi', 'fprem', 'fptan', 'fprem1',
         'rdmsr', 'getsec', 'xaddq', 'vstr', 'ldmdb', 'sbc', 'bge', 'cmpmi', 'ldrexb', 'stc2l', 'blt', 'trap', 'vmrs',
         'mla', 'strmi', 'stcl', 'ldrb', 'vmsr', 'orns', 'tbb', 'cbz', 'lslmi', 'ble', 'beq', 'sbcs', 'sxth', 'movgt',
         'bvc', 'rsbs', 'bics', 'bls', 'asrvc', 'adcs', 'ldrd', 'addmi', 'blx', 'bic', 'uxtb', 'clz', 'orrhs', 'eors',
         'bhs', 'uxth', 'ldrex', 'mvns', 'cdp2', 'ldrhmi', 'asrmi', 'ldr', 'ldrsb', 'bl', 'tbh', 'mvn', 'strb', 'asrs',
         'rsbhs', 'hint', 'ldrmi', 'bpl', 'movne', 'bgt', 'movt', 'bfi', 'stm', 'ldm', 'bkpt', 'ubfx', 'adc', 'cdp',
         'stmdb', 'strex', 'udf', 'bx', 'eor', 'blo', 'cmn', '__brkdiv0', 'revsh', 'adr', 'bmi', 'movge', 'svc', 'bne',
         'rsbmi', 'cbnz', 'bhi', 'mov', 'ldcl', 'rev', 'pkhbt', 'movhi', 'rsb', 'sxtb', 'stc2', 'bfc', 'sbfx', 'bvs',
         'lsrmi', 'ldc', 'dmb', 'ldc2l', 'sev', 'strh', 'cmp', 'strexb', 'tst', 'ldc2', 'pophi', 'rev16', 'setnp',
         'vmovntps', 'pmovzxbd', 'smsww', 'vdivss', 'lgdtq', 'cmpsw', 'inw', 'lidtq', 'sidtq', 'lmsww', 'vpminsw',
         'fdecstp', 'cwtd', 'movntdq', 'lddqu', 'sfence', 'pmaxub', 'pminub', 'xgetbv', 'vucomisd', 'vzeroupper',
         'movmskps', 'jmpw', 'sidtl', 'orps', 'leaw', 'lesw', 'xabort', 'ldsw', 'jcxz', 'minps', 'vmwritel', 'popfw',
         'ud1l', 'scasw', 'sysexitl', 'movntil', 'cmovbw', 'vpsadbw', 'fninit', 'aesimc', 'pclmulqdq', 'aesenclast',
         'aesenc', 'aesdec', 'aesdeclast', 'aeskeygenassist', 'pinsrd', 'vpmulhw', 'vmovddup', 'vpblendmd', 'xrstor',
         'cmovgw', 'fxsave', 'vminps', 'sqrtpd', 'movupd', 'divpd', 'movmskpd', 'bsfq', 'fxrstor', 'xsave', 'bsrq',
         'maskmovdqu', 'psignb', 'movdq2q', 'pabsb', 'movq2dq', 'mfence', 'btcq', 'pmulhrw', 'vmreadq', 'vpermilpd',
         'vpsraw', 'vmwriteq', 'sysretq', 'vmulps', 'vorpd', 'vorps', 'vmptrld', 'pfrcpit1', 'pfrsqit1', 'pfpnacc',
         'pfnacc', 'pfrcp', 'pfrcpit2', 'pfacc', 'rcpss', 'pi2fd', 'pf2id', 'strl', 'movhpd', 'vcmppd', 'vpinsrw',
         'vblendmps', 'shrdq', 'vpblendmb', 'vlddqu', 'vpsllw', 'vpsrlw', 'xrstors', 'vfnmsubps', 'pi2fw', 'rdrandq',
         'wrgsbaseq', 'maxsd', 'minsd', 'tzcntl', 'lfence', 'monitor', 'mwait', 'vfmsub213sd', 'vpsubq', 'vfmsub213ss',
         'vpsrad', 'vcmpss', 'stmhs', 'stmge', 'stmeq', 'vp4dpwssds', 'vpmuludq', 'vpackuswb', 'vminpd', 'ljmpq',
         'smswl', 'vpmovzxbw', 'vpminub', 'vmaxpd', 'vpermi2d', 'cmpnlesd'],
    0: ['lretl', 'retl', 'iretl', 'lretw', 'lretq'],
    1: ['sbbb', 'subl', 'sbbl', 'subb', 'sbbq', 'fsubl', 'fsubs', 'fsubp', 'fsub', 'fsubr', 'fisubrs', 'subw',
        'fisubrl', 'fisubl', 'sbbw', 'fsubrs', 'psubsb', 'fisubs', 'fsubrl', 'psubb', 'bsfl', 'ss', 'vsubsd', 'subsd',
        'subss', 'fsubrp', 'psubw', 'psubsw', 'psubusb', 'subps', 'psubusw', 'sub', 'b', 'subs', 'pfsub', 'pshufb',
        'pfsubr', 'subls', 'vfmsubps', 'vpsubusw', 'vsubss', 'vsubps'],
    2: ['movl', 'movsb', 'movb', 'movw', 'movsl', 'movaps', 'movsd', 'movsbq', 'cmovneq', 'movq', 'movswl', 'movabsq',
        'movslq', 'cmovll', 'movswq', 'movsbl', 'movsbw', 'movsw', 'movdqa', 'cmovnel', 'movhps', 'movlps', 'movdqu',
        'cmovsq', 'movd', 'cmovsl', 'cmovoq', 'cmovsw', 'movabsb', 'movlhps', 'movapd', 'movss', 'cmovol', 'cmovnol',
        'cmovnsl', 'vmovsd', 'vmovq', 'vmovdqa', 'cmovnsq', 'cmovnpl', 'movabsl', 'vmovdqu', 'movlt', 'movhs', 'moveq',
        'movs', 'movlo', 'movls', 'vmov', 'movle', 'vmovaps', 'vmovups', 'movsq', 'movhlps', 'vmovapd', 'cmovnoq',
        'movabsw', 'vmovd', 'vmovlps'],
    3: ['daa', 'aaa', 'aas', 'aad', 'aam'],
    4: ['lcalll', 'shll', 'calll', 'shrl', 'callq', 'shrdl', 'lesl', 'lodsl', 'ldsl', 'lsll', 'shldl', 'psllq', 'sldtl',
        'pslld', 'callw', 'lsls', 'ldrhlo', 'lsrs', 'lsr', 'lsrls', 'ldrsh', 'lsl', 'lslls', 'lslvs', 'lsrpl', 'lslhs',
        'lslhi', 'ldrhvs', 'lsrlo', 'ldrh', 'lfsl', 'lcallq', 'lssl', 'lcallw', 'lgsl', 'lslq', 'ldrhls', 'lsrhi'],
    5: ['fdivr', 'fdivl', 'divl', 'idivl', 'fidivrs', 'fdivrs', 'fdivrp', 'fidivl', 'fidivrl', 'fdiv', 'fidivs', 'fdivs', 'fdivrl', 'fdivp'],
    6: ['sete', 'setne', 'setge', 'setle', 'setbe', 'setae'],
    7: ['imull', 'mull', 'fmul', 'fmuls', 'fmulp', 'fimull', 'fmull', 'mul', 'umlal', 'smull', 'umull', 'pfmul'],
    8: ['cmoveq', 'cmovel', 'cmovaeq', 'cmovbel', 'cmovlq', 'cmovlel', 'cmovgel', 'cmovael', 'cmovgeq', 'cmovbeq', 'cmovleq'],
    9: ['prefetchnta', 'prefetchw', 'prefetch', 'prefetcht0', 'prefetchwt1', 'prefetcht2', 'prefetcht1'],
    10: ['addl', 'addb', 'das', 'addq', 'addw', 'fadds', 'faddp', 'fadd', 'fiadds', 'faddl', 'paddw', 'addps', 'addpd',
         'subpd', 'addsd', 'andpd', 'paddd', 'vaddsd', 'pand', 'vpand', 'paddusb', 'addss', 'paddusw', 'vhsubpd',
         'paddsb', 'andps', 'psubd', 'ds', 'pandn', 'paddsw', 'vpaddd', 'pmaddwd', 'paddb', 'andnps', 'vpaddusb',
         'phaddsw', 'paddq', 'andnpd', 'adds', 'and', 'add', 'ands', 'vaddss', 'vaddpd', 'pfadd', 'vpaddq', 'vandpd',
         'vhaddps', 'vaddsubpd', 'vpaddw', 'pmaddubsw', 'vandnpd', 'vandnps', 'vpsubd', 'vpaddusw', 'vaddps', 'pswapd',
         'vfmaddps', 'vsubpd', 'haddps', 'vandps'],
    11: ['ficoml', 'fucomp', 'fcomp', 'fcom', 'fcompl', 'fucomi', 'fucom', 'fcomps', 'fcoml', 'fcoms', 'ficoms', 'fcmovu', 'ficomps', 'ficompl', 'fucompp', 'fcompp', 'fcompi', 'fcos', 'fucompi', 'fcomi'],
    12: ['fstps', 'fstp', 'fstpl', 'fstpt', 'fistl', 'fsts', 'ftst', 'fisttpll', 'fstl', 'fistps', 'fistpl', 'fisttps', 'fistpll', 'fisttpl', 'fists', 'fbstp'],
    13: ['cmpl', 'cmpsl', 'pcmpeqw', 'pcmpeqb', 'cmpeqsd', 'pcmpeqd', 'cmpsq', 'cmpltpd', 'cmpps', 'cmpeqps', 'cmpltss',
         'cmpnltps', 'cmpltps', 'pfcmpge', 'cmpneqps', 'cmpleps', 'pfcmpeq', 'vpcmpeqw', 'cmpss', 'cmpless', 'cmpnltsd',
         'cmplesd', 'cmpltsd', 'stmpl', 'vcmpps'],
    14: ['pushl', 'pushfl', 'pushq', 'pushw', 'shufps', 'pshuflw', 'pshufhw', 'pushfw', 'pshufw', 'push', 'vpush',
         'pushaw', 'vpshufhw', 'vpshuflw'],
    15: ['punpckhwd', 'unpcklps', 'unpckhpd', 'punpcklwd', 'unpcklpd', 'punpckldq', 'punpckhbw', 'punpcklbw',
         'punpckhdq', 'unpckhps', 'punpcklqdq', 'punpckhqdq', 'vpunpcklqdq', 'vpunpcklwd', 'vunpckhpd', 'vpunpckhwd',
         'vpunpckhdq'],
    16: ['fldln2', 'fldt', 'fldl2t', 'fldl2e', 'fldlg2'],
    17: ['cvttps2pi', 'cvtdq2ps', 'cvtps2pd', 'cvtdq2pd', 'vcvtdq2pd', 'cvtpd2ps', 'cvtps2dq', 'cvtps2pi', 'cvtpi2ps',
         'cvttpd2dq', 'cvttps2dq', 'cvtpi2pd', 'vcvtdq2ps', 'vcvtpd2dq', 'cvtpd2dq'],
    18: ['cmovew', 'cmovnew', 'cmovaew', 'cmovgew', 'cmovbew'],
    19: ['pcmpgtb', 'pcmpgtd', 'pcmpgtw', 'pfcmpgt', 'vpcmpgtb'],
    20: ['vfmadd213sd', 'vfmadd231sd', 'vfnmadd132sd', 'vfnmadd231sd', 'vfmadd213ss'],
    21: ['cvtsi2sd', 'cvtss2sd', 'cvtsi2ss', 'cvttsd2si', 'cvtsd2ss', 'cvttss2si', 'cvtsi2sdl', 'cvtsd2si', 'cvtsi2ssl',
         'cvtsi2sdq', 'cvtsi2ssq', 'cvtss2si', 'vcvtsd2ss', 'vcvtsi2ssl', 'vcvtss2sd'],
    22: ['sqrtps', 'rsqrtps', 'sqrtsd', 'str', 'strd', 'vrsqrtps', 'pfrsqrt', 'rsqrtss', 'sqrtss'],
    23: ['pxor', 'vpor', 'vxorps', 'vpxor', 'vxorpd'],
    24: ['cmpxchgl', 'cmpxchgq', 'cmpxchgb', 'cmpxchg8b', 'cmpxchg16b'],
    25: ['itett', 'ittee', 'itte', 'iteet', 'itt', 'ite', 'iteee', 'it', 'itete', 'ittte', 'ittt', 'itee', 'ittet',
         'itet'],
    26: ['mcr2', 'mcr', 'mrrc2', 'mrc', 'mrc2', 'mcrr', 'mcrr2'],
    27: ['vldr', 'ldrvs', 'vpsrld', 'vpsrlvd'],
    28: ['mulsd', 'mulps', 'mulss', 'muls', 'vmulss'],
    29: ['rorb', 'rorl', 'rorw', 'rorq', 'rors', 'orr', 'orrs'],
    30: ['popl', 'popq', 'popw', 'vpop', 'pop'],
    31: ['pshufd', 'shufpd', 'vshufps', 'vshufpd', 'vpshufb', 'vpshufd'],
    32: ['ucomiss', 'vcomisd', 'comiss', 'vcomiss'],
}
#
# DATA = {-1: ['insl', 'jbe', 'outsb', 'jnp', 'insb', 'jne', 'leave', 'loop', 'outb', 'je', 'stc', 'jge', 'outl', 'ja', 'xorb', 'retw', 'js', 'xchgl', 'lahf', 'imulw', 'pushal', 'jno', 'cmc', 'cli', 'jle', 'outsl', 'jo', 'arpl', 'jns', 'bound', 'jp', 'orb', 'incl', 'jb', 'jae', 'rclb', 'nop', 'seto', 'jl', 'movzbl', 'rep', 'loope', 'jg', 'lock', 'notl', 'rcrb', 'negl', 'loopne', 'cmpsb', 'repne', 'inl', 'movzwl', 'xchgb', 'lodsb', 'inb', 'xlatb', 'orw', 'movups', 'bswapq', 'movntiq', 'retq', 'xorq', 'nopl', 'negw', 'shrq', 'cltq', 'notq', 'incb', 'leaq', 'incq', 'shlq', 'jmpq', 'xorw', 'sarq', 'decq', 'imulq', 'orq', 'negq', 'nopw', 'notb', 'cpuid', 'incw', 'fchs', 'stmxcsr', 'wait', 'fxch', 'decb', 'ldmxcsr', 'rcrl', 'negb', 'movzbw', 'fnstcw', 'fldcw', 'sarw', 'shlw', 'filds', 'fsqrt', 'xbegin', 'fxtract', 'ffreep', 'xchgw', 'rcrw', 'fsincos', 'xacquire', 'clc', 'jecxz', 'cwtl', 'fldenv', 'imulb', 'fyl2xp1', 'fnstenv', 'decw', 'hlt', 'xrelease', 'sahf', 'rclw', 'notw', 'fnsave', 'fscale', 'sldtw', 'shrw', 'pmuludq', 'clts', 'sgdtl', 'wbinvd', 'outsw', 'pavgb', 'cldemote', 'pcmpistri', 'btq', 'movzwq', 'movzbq', 'sets', 'cqto', 'maxps', 'rdpmc', 'packsswb', 'setp', 'data16', 'rex64', 'gs', 'invd', 'jrcxz', 'fnclex', 'pause', 'fabs', 'fyl2x', 'f2xm1', 'pextrw', 'mulpd', 'frndint', 'palignr', 'fxam', 'movlpd', 'cbtw', 'psrlq', 'verw', 'invlpg', 'vpsrlq', 'btsq', 'xchgq', 'btrq', 'movntps', 'bswapl', 'pavgw', 'ltrw', 'divps', 'femms', 'emms', 'fnop', 'ud2', 'verr', 'divss', 'cmpnlepd', 'outw', 'insw', 'vpsravd', 'pinsrw', 'psraw', 'psrad', 'psrld', 'psrlw', 'crc32l', 'packuswb', 'pminsw', 'pmaxsw', 'movntq', 'btw', 'sgdtq', 'packssdw', 'lodsq', 'rdseedl', 'rclq', 'rcrq', 'vmulpd', 'iretq', 'bndldx', 'fpatan', 'fincstp', 'bndstx', 'ffree', 'vmreadl', 'lodsw', 'iretw', 'setno', 'fsin', 'fprem1', 'fptan', 'fprem', 'fldpi', 'cmn', 'bpl', 'bvc', 'rev16', 'mvn', 'bics', 'beq', 'orns', 'ldmdb', 'bge', 'bvs', 'cbnz', 'svc', 'sbfx', 'bhi', 'dmb', 'eors', 'adr', 'cdp', 'ldrmi', 'bfc', 'ldrexb', 'asrvc', 'bne', 'ldrex', 'sev', 'movgt', 'tbh', 'asrmi', 'bgt', 'pophi', 'addmi', 'ldm', 'cbz', 'movhi', 'uxth', 'bmi', 'sxth', 'bl', '__brkdiv0', 'bfi', 'lsrmi', 'stmdb', 'stcl', 'trap', 'asrs', 'mvns', 'bic', 'clz', 'stm', 'eor', 'stc2l', 'revsh', 'movne', 'cdp2', 'udf', 'rsbmi', 'uxtb', 'ble', 'pkhbt', 'stc2', 'sxtb', 'bx', 'movge', 'bkpt', 'ldrhmi', 'ubfx', 'tbb', 'strex', 'strexb', 'blx', 'strmi', 'blo', 'rev', 'setnp', 'vmovntps', 'pmovzxbd', 'fdecstp', 'lgdtq', 'vdivss', 'lidtq', 'inw', 'sidtq', 'smsww', 'lmsww', 'vpminsw', 'cmpsw', 'cwtd', 'lddqu', 'pminub', 'sfence', 'pmaxub', 'movntdq', 'xgetbv', 'vzeroupper', 'jmpw', 'sidtl', 'vmwritel', 'jcxz', 'leaw', 'ud1l', 'ldsw', 'xabort', 'minps', 'lesw', 'sysexitl', 'movntil', 'pclmulqdq', 'aesimc', 'pinsrd', 'aeskeygenassist', 'vpblendmd', 'fxsave', 'vminps', 'bsfq', 'xsave', 'movdq2q', 'psignb', 'movq2dq', 'maskmovdqu', 'mfence', 'btcq', 'vmwriteq', 'vpermilpd', 'vpsraw', 'vmreadq', 'vmptrld', 'pi2fd', 'pfnacc', 'pfrcpit1', 'pfrsqit1', 'pfpnacc', 'pfrcp', 'pfrcpit2', 'pf2id', 'pfacc', 'movhpd', 'vpinsrw', 'vcmppd', 'vblendmps', 'shrdq', 'vpblendmb', 'vpsrlw', 'vlddqu', 'pi2fw', 'wrgsbaseq', 'rdrandq', 'tzcntl', 'minsd', 'maxsd', 'lfence', 'mwait', 'monitor', 'vfmsub213ss', 'stmhs', 'stmeq', 'stmge', 'vp4dpwssds', 'vpackuswb', 'vminpd', 'vpmuludq', 'vpmovzxbw', 'vpminub', 'vpermi2d', 'vmaxpd'],
#         0: ['movsb', 'movl', 'movb', 'movsl', 'movw', 'movaps', 'movswl', 'movswq', 'movabsq', 'cmoveq', 'cmovll', 'cmoval', 'cmovneq', 'movslq', 'movq', 'movsbl', 'cmovel', 'cmovgl', 'movsd', 'cmovbq', 'cmovaeq', 'movsbq', 'movsw', 'movdqa', 'movsbw', 'fcmovnb', 'cmovbel', 'fcmove', 'cmovnel', 'fcmovb', 'fcmovbe', 'fcmovnu', 'movlps', 'movhps', 'cmovoq', 'movdqu', 'movd', 'pmovmskb', 'cmovsq', 'cmovbl', 'cmovsl', 'cmovaq', 'cmovlel', 'cmovew', 'cmovlq', 'cmovgq', 'cmovsw', 'cmovnew', 'movabsb', 'cmovgel', 'movlhps', 'cmovael', 'movss', 'movapd', 'cmovol', 'cmovnol', 'cmovnsl', 'vpmovmskb', 'vmovsd', 'vmovdqa', 'vmovq', 'cmovgeq', 'cmovnsq', 'fcmovne', 'fcmovnbe', 'cmovnpl', 'movabsl', 'vmovdqu', 'cmovlw', 'cmovbeq', 'cmovaw', 'cmovpl', 'cmovleq', 'vmov', 'movls', 'movlt', 'moveq', 'movhs', 'mov', 'movt', 'movs', 'movlo', 'movle', 'vmovaps', 'vmovups', 'movsq', 'cmovaew', 'movhlps', 'vmovapd', 'movmskps', 'cmovbw', 'vmovddup', 'cmovgw', 'cmovgew', 'movmskpd', 'movupd', 'cmovnoq', 'movabsw', 'vmovd', 'vmovlps', 'cmovbew'],
#         1: ['orl', 'subb', 'andl', 'xorl', 'pushl', 'addb', 'addl', 'sbbb', 'sbbl', 'subl', 'salc', 'decl', 'scasb', 'leal', 'andb', 'imull', 'adcl', 'calll', 'cld', 'rolb', 'shrl', 'xaddl', 'cltd', 'lcalll', 'shll', 'das', 'pushfl', 'roll', 'sarl', 'mull', 'sarb', 'callq', 'pushq', 'subq', 'sbbq', 'addq', 'andq', 'andw', 'btsl', 'addw', 'fmul', 'fldz', 'fldl', 'fadds', 'flds', 'shlb', 'rorl', 'shrdl', 'lesl', 'ldsl', 'fisubl', 'fiaddl', 'fisubrs', 'sbbw', 'fbld', 'fimull', 'shrb', 'subw', 'fldl2t', 'fildll', 'fisubs', 'fmuls', 'faddp', 'fsubr', 'fldt', 'fmulp', 'psubsb', 'fisubrl', 'fld', 'faddl', 'fmull', 'pushw', 'fsub', 'fld1', 'fadd', 'mulb', 'fsubrs', 'rolw', 'fldln2', 'rcll', 'fildl', 'fsubl', 'xaddb', 'fsubp', 'mulw', 'scasl', 'fiadds', 'fsubs', 'fimuls', 'lodsl', 'fldl2e', 'paddw', 'shufps', 'lsll', 'fsubrl', 'psubb', 'larl', 'bsrl', 'mulq', 'bsfl', 'pshuflw', 'pshufd', 'mulsd', 'psrldq', 'divsd', 'mulps', 'addps', 'syscall', 'cs', 'pushfq', 'ss', 'shldl', 'addsd', 'psllq', 'subpd', 'addpd', 'mulss', 'andpd', 'paddd', 'lgdtl', 'vpand', 'vmulsd', 'pand', 'vaddsd', 'vsubsd', 'subsd', 'paddusb', 'subss', 'addss', 'vhsubpd', 'paddusw', 'lidtl', 'paddsb', 'fsubrp', 'lldtw', 'andps', 'sldtl', 'fldlg2', 'psubd', 'pslld', 'ds', 'rolq', 'psllw', 'pmaddwd', 'vpaddd', 'pslldq', 'psadbw', 'psubw', 'pmullw', 'paddsw', 'psubsw', 'pandn', 'pshufhw', 'paddb', 'andnps', 'psubusb', 'psubq', 'pshufw', 'pushfw', 'subps', 'scasq', 'vpaddusb', 'psubusw', 'phaddsw', 'callw', 'paddq', 'andnpd', 'shufpd', 'xaddq', 'muls', 'smull', 'lsrpl', 'vpush', 'b', 'lsrls', 'ands', 'sub', 'rsb', 'ldrvs', 'and', 'adds', 'ldrh', 'lslhi', 'rsbhs', 'ldrb', 'umlal', 'subs', 'lsrlo', 'lslhs', 'adcs', 'ldrsh', 'lslmi', 'sbc', 'ldrd', 'umull', 'add', 'strb', 'bhs', 'rsbs', 'ldcl', 'vldr', 'sbcs', 'ldrhvs', 'ldc2', 'ldc', 'lsls', 'lsrs', 'ldrhlo', 'ldr', 'ldrsb', 'lslvs', 'lslls', 'mul', 'push', 'mla', 'ldc2l', 'lsr', 'lsl', 'bls', 'lssl', 'lcallq', 'vaddpd', 'vaddss', 'lfsl', 'pfmul', 'pfsub', 'pfadd', 'vpaddq', 'lcallw', 'vshufpd', 'scasw', 'pushaw', 'vshufps', 'vandpd', 'lgsl', 'vpsadbw', 'pshufb', 'vpshufb', 'vhaddps', 'vaddsubpd', 'vpaddw', 'divpd', 'bsrq', 'pabsb', 'pmaddubsw', 'lslq', 'vmulss', 'vpaddusw', 'vpsubd', 'vandnpd', 'vandnps', 'vaddps', 'vmulps', 'pswapd', 'pfsubr', 'strl', 'lsrhi', 'subls', 'ldrhls', 'vfmsubps', 'vpshufd', 'vpshuflw', 'vpsllw', 'vpshufhw', 'vfmaddps', 'vsubpd', 'vfnmsubps', 'vpsubusw', 'haddps', 'vandps', 'vsubss', 'vpsrld', 'vpsubq', 'vpsrad', 'vsubps', 'smswl', 'vpsrlvd'],
#         12: ['cmpl', 'cmpb', 'cmpw', 'cmpq', 'cmpsl', 'pcmpeqb', 'pcmpeqw', 'rcpps', 'cmpeqsd', 'pcmpeqd', 'cmpsq', 'cmpltpd', 'cmpmi', 'cmp', 'cmpps', 'cmpeqps', 'cmpltss', 'cmpltps', 'cmpnltps', 'cmpneqps', 'pfcmpge', 'cmpleps', 'rcpss', 'pfcmpeq', 'vpcmpeqw', 'cmpss', 'cmpless', 'cmpltsd', 'cmpnltsd', 'cmplesd', 'vcmpss', 'stmpl', 'vcmpps', 'cmpnlesd'],
#         5: ['popl', 'popal', 'popfl', 'popw', 'popaw', 'popfq', 'popfw'],
#         3: ['std', 'enter', 'testl', 'testb', 'rorb', 'sete', 'stosl', 'testw', 'setne', 'testq', 'setg', 'setl', 'stosw', 'setge', 'stosb', 'frstor', 'setle', 'rorw', 'strw', 'sysretl', 'sysenter', 'seta', 'setns', 'setb', 'rorq', 'setbe', 'setae', 'sqrtps', 'rdtsc', 'es', 'stosq', 'rsqrtps', 'sqrtsd', 'getsec', 'iteet', 'rors', 'itte', 'itete', 'vstr', 'str', 'iteee', 'strh', 'itett', 'strd', 'it', 'ite', 'itt', 'orr', 'orrhs', 'ittte', 'orrs', 'ittee', 'vrsqrtps', 'aesenc', 'aesenclast', 'aesdeclast', 'aesdec', 'xrstor', 'sqrtpd', 'fxrstor', 'sysretq', 'pfrsqrt', 'rsqrtss', 'sqrtss', 'xrstors', 'itee', 'ittet', 'ittt', 'itet'],
#         2: ['aaa', 'daa', 'aas', 'aad', 'aam'],
#         4: ['lretl', 'retl', 'iretl', 'lretw', 'lretq'],
#         23: ['adcb', 'adcw', 'adcq', 'adc'],
#         13: ['int3', 'int', 'into', 'hint', 'fninit'],
#         7: ['jmp', 'jmpl', 'ljmpl', 'ljmpw', 'ljmpq'],
#         10: ['ficoml', 'fcompl', 'fcomp', 'fucomp', 'fcom', 'fucompp', 'fcomps', 'fucomi', 'fcoml', 'fcoms', 'ficompl', 'ficoms', 'fcmovu', 'ficomps', 'fucom', 'ucomisd', 'comisd', 'ucomiss', 'fcompp', 'fcompi', 'vcomisd', 'fcos', 'fucompi', 'comiss', 'fcomi', 'vucomisd', 'vcomiss'],
#         21: ['cmpxchgl', 'cmpxchgq', 'cmpxchgb', 'cmpxchg8b', 'cmpxchg16b'],
#         6: ['fdivr', 'divq', 'divl', 'fdivl', 'idivl', 'idivw', 'fdiv', 'fidivrl', 'fdivrp', 'divw', 'divb', 'fdivrs', 'fdivs', 'fidivl', 'fidivrs', 'fidivs', 'idivb', 'fdivrl', 'idivq', 'fdivp'],
#         8: ['btrl', 'btl', 'btcl', 'blt'],
#         9: ['prefetchnta', 'prefetchw', 'prefetch', 'prefetcht0', 'prefetchwt1', 'prefetcht2', 'prefetcht1'],
#         24: ['popq', 'vpop', 'pop'],
#         11: ['fstpl', 'fnstsw', 'fstp', 'fstps', 'fistps', 'fisttpll', 'fists', 'ftst', 'fisttps', 'fistpl', 'fsts', 'sti', 'fistpll', 'fstl', 'fistl', 'fisttpl', 'fstpt', 'fbstp', 'fs', 'fst', 'tst'],
#         14: ['punpckhwd', 'unpcklps', 'unpckhpd', 'punpcklwd', 'unpcklpd', 'punpckldq', 'punpcklbw', 'punpckhdq', 'punpckhbw', 'unpckhps', 'punpcklqdq', 'punpckhqdq', 'vpunpcklqdq', 'vpunpcklwd', 'vpunpckhdq', 'vpunpckhwd', 'vunpckhpd'],
#         15: ['cvttps2pi', 'cvtdq2ps', 'cvtdq2pd', 'cvtps2pd', 'vcvtdq2pd', 'cvtsi2sd', 'cvttsd2si', 'cvtpd2ps', 'cvtsi2ss', 'cvtss2sd', 'cvtsd2ss', 'cvtps2dq', 'cvttss2si', 'cvtps2pi', 'cvtpi2ps', 'cvtsi2sdl', 'cvtsi2ssl', 'cvtsd2si', 'cvtsi2sdq', 'cvtsi2ssq', 'cvtss2si', 'cvttpd2dq', 'vcvtsd2ss', 'vcvtss2sd', 'vcvtsi2ssl', 'cvttps2dq', 'cvtpi2pd', 'cvtpd2dq', 'vcvtpd2dq', 'vcvtdq2ps'],
#         16: ['rsm', 'wrmsr', 'rdmsr', 'vmsr', 'vmrs'],
#         17: ['xorps', 'por', 'orpd', 'xorpd', 'pxor', 'vxorps', 'vpor', 'vpxor', 'orps', 'vorpd', 'vxorpd', 'vorps'],
#         18: ['pcmpgtb', 'pcmpgtd', 'pcmpgtw', 'pfcmpgt', 'vpcmpgtb'],
#         20: ['pmulhuw', 'pmulhw', 'vpmulhw', 'pmulhrw'],
#         19: ['vfmadd213sd', 'vfmadd231sd', 'vfmsub213sd', 'vfnmadd132sd', 'vfnmadd231sd', 'vfmadd213ss'],
#         22: ['mcr2', 'mrrc2', 'mcrr', 'mcr', 'mcrr2', 'mrc2', 'mrc']}


REVERSED_MAP = {'lretl': 0, 'retl': 0, 'iretl': 0, 'lretw': 0, 'lretq': 0, 'sbbb': 1, 'subl': 1, 'sbbl': 1, 'subb': 1, 'sbbq': 1, 'fsubl': 1, 'fsubs': 1, 'fsubp': 1, 'fsub': 1, 'fsubr': 1, 'fisubrs': 1, 'subw': 1, 'fisubrl': 1, 'fisubl': 1, 'sbbw': 1, 'fsubrs': 1, 'psubsb': 1, 'fisubs': 1, 'fsubrl': 1, 'psubb': 1, 'bsfl': 1, 'ss': 1, 'vsubsd': 1, 'subsd': 1, 'subss': 1, 'fsubrp': 1, 'psubw': 1, 'psubsw': 1, 'psubusb': 1, 'subps': 1, 'psubusw': 1, 'sub': 1, 'b': 1, 'subs': 1, 'pfsub': 1, 'pshufb': 1, 'pfsubr': 1, 'subls': 1, 'vfmsubps': 1, 'vpsubusw': 1, 'vsubss': 1, 'vsubps': 1, 'movl': 2, 'movsb': 2, 'movb': 2, 'movw': 2, 'movsl': 2, 'movaps': 2, 'movsd': 2, 'movsbq': 2, 'cmovneq': 2, 'movq': 2, 'movswl': 2, 'movabsq': 2, 'movslq': 2, 'cmovll': 2, 'movswq': 2, 'movsbl': 2, 'movsbw': 2, 'movsw': 2, 'movdqa': 2, 'cmovnel': 2, 'movhps': 2, 'movlps': 2, 'movdqu': 2, 'cmovsq': 2, 'movd': 2, 'cmovsl': 2, 'cmovoq': 2, 'cmovsw': 2, 'movabsb': 2, 'movlhps': 2, 'movapd': 2, 'movss': 2, 'cmovol': 2, 'cmovnol': 2, 'cmovnsl': 2, 'vmovsd': 2, 'vmovq': 2, 'vmovdqa': 2, 'cmovnsq': 2, 'cmovnpl': 2, 'movabsl': 2, 'vmovdqu': 2, 'movlt': 2, 'movhs': 2, 'moveq': 2, 'movs': 2, 'movlo': 2, 'movls': 2, 'vmov': 2, 'movle': 2, 'vmovaps': 2, 'vmovups': 2, 'movsq': 2, 'movhlps': 2, 'vmovapd': 2, 'cmovnoq': 2, 'movabsw': 2, 'vmovd': 2, 'vmovlps': 2, 'daa': 3, 'aaa': 3, 'aas': 3, 'aad': 3, 'aam': 3, 'lcalll': 4, 'shll': 4, 'calll': 4, 'shrl': 4, 'callq': 4, 'shrdl': 4, 'lesl': 4, 'lodsl': 4, 'ldsl': 4, 'lsll': 4, 'shldl': 4, 'psllq': 4, 'sldtl': 4, 'pslld': 4, 'callw': 4, 'lsls': 4, 'ldrhlo': 4, 'lsrs': 4, 'lsr': 4, 'lsrls': 4, 'ldrsh': 4, 'lsl': 4, 'lslls': 4, 'lslvs': 4, 'lsrpl': 4, 'lslhs': 4, 'lslhi': 4, 'ldrhvs': 4, 'lsrlo': 4, 'ldrh': 4, 'lfsl': 4, 'lcallq': 4, 'lssl': 4, 'lcallw': 4, 'lgsl': 4, 'lslq': 4, 'ldrhls': 4, 'lsrhi': 4, 'fdivr': 5, 'fdivl': 5, 'divl': 5, 'idivl': 5, 'fidivrs': 5, 'fdivrs': 5, 'fdivrp': 5, 'fidivl': 5, 'fidivrl': 5, 'fdiv': 5, 'fidivs': 5, 'fdivs': 5, 'fdivrl': 5, 'fdivp': 5, 'sete': 6, 'setne': 6, 'setge': 6, 'setle': 6, 'setbe': 6, 'setae': 6, 'imull': 7, 'mull': 7, 'fmul': 7, 'fmuls': 7, 'fmulp': 7, 'fimull': 7, 'fmull': 7, 'mul': 7, 'umlal': 7, 'smull': 7, 'umull': 7, 'pfmul': 7, 'cmoveq': 8, 'cmovel': 8, 'cmovaeq': 8, 'cmovbel': 8, 'cmovlq': 8, 'cmovlel': 8, 'cmovgel': 8, 'cmovael': 8, 'cmovgeq': 8, 'cmovbeq': 8, 'cmovleq': 8, 'prefetchnta': 9, 'prefetchw': 9, 'prefetch': 9, 'prefetcht0': 9, 'prefetchwt1': 9, 'prefetcht2': 9, 'prefetcht1': 9, 'addl': 10, 'addb': 10, 'das': 10, 'addq': 10, 'addw': 10, 'fadds': 10, 'faddp': 10, 'fadd': 10, 'fiadds': 10, 'faddl': 10, 'paddw': 10, 'addps': 10, 'addpd': 10, 'subpd': 10, 'addsd': 10, 'andpd': 10, 'paddd': 10, 'vaddsd': 10, 'pand': 10, 'vpand': 10, 'paddusb': 10, 'addss': 10, 'paddusw': 10, 'vhsubpd': 10, 'paddsb': 10, 'andps': 10, 'psubd': 10, 'ds': 10, 'pandn': 10, 'paddsw': 10, 'vpaddd': 10, 'pmaddwd': 10, 'paddb': 10, 'andnps': 10, 'vpaddusb': 10, 'phaddsw': 10, 'paddq': 10, 'andnpd': 10, 'adds': 10, 'and': 10, 'add': 10, 'ands': 10, 'vaddss': 10, 'vaddpd': 10, 'pfadd': 10, 'vpaddq': 10, 'vandpd': 10, 'vhaddps': 10, 'vaddsubpd': 10, 'vpaddw': 10, 'pmaddubsw': 10, 'vandnpd': 10, 'vandnps': 10, 'vpsubd': 10, 'vpaddusw': 10, 'vaddps': 10, 'pswapd': 10, 'vfmaddps': 10, 'vsubpd': 10, 'haddps': 10, 'vandps': 10, 'ficoml': 11, 'fucomp': 11, 'fcomp': 11, 'fcom': 11, 'fcompl': 11, 'fucomi': 11, 'fucom': 11, 'fcomps': 11, 'fcoml': 11, 'fcoms': 11, 'ficoms': 11, 'fcmovu': 11, 'ficomps': 11, 'ficompl': 11, 'fucompp': 11, 'fcompp': 11, 'fcompi': 11, 'fcos': 11, 'fucompi': 11, 'fcomi': 11, 'fstps': 12, 'fstp': 12, 'fstpl': 12, 'fstpt': 12, 'fistl': 12, 'fsts': 12, 'ftst': 12, 'fisttpll': 12, 'fstl': 12, 'fistps': 12, 'fistpl': 12, 'fisttps': 12, 'fistpll': 12, 'fisttpl': 12, 'fists': 12, 'fbstp': 12, 'cmpl': 13, 'cmpsl': 13, 'pcmpeqw': 13, 'pcmpeqb': 13, 'cmpeqsd': 13, 'pcmpeqd': 13, 'cmpsq': 13, 'cmpltpd': 13, 'cmpps': 13, 'cmpeqps': 13, 'cmpltss': 13, 'cmpnltps': 13, 'cmpltps': 13, 'pfcmpge': 13, 'cmpneqps': 13, 'cmpleps': 13, 'pfcmpeq': 13, 'vpcmpeqw': 13, 'cmpss': 13, 'cmpless': 13, 'cmpnltsd': 13, 'cmplesd': 13, 'cmpltsd': 13, 'stmpl': 13, 'vcmpps': 13, 'pushl': 14, 'pushfl': 14, 'pushq': 14, 'pushw': 14, 'shufps': 14, 'pshuflw': 14, 'pshufhw': 14, 'pushfw': 14, 'pshufw': 14, 'push': 14, 'vpush': 14, 'pushaw': 14, 'vpshufhw': 14, 'vpshuflw': 14, 'punpckhwd': 15, 'unpcklps': 15, 'unpckhpd': 15, 'punpcklwd': 15, 'unpcklpd': 15, 'punpckldq': 15, 'punpckhbw': 15, 'punpcklbw': 15, 'punpckhdq': 15, 'unpckhps': 15, 'punpcklqdq': 15, 'punpckhqdq': 15, 'vpunpcklqdq': 15, 'vpunpcklwd': 15, 'vunpckhpd': 15, 'vpunpckhwd': 15, 'vpunpckhdq': 15, 'fldln2': 16, 'fldt': 16, 'fldl2t': 16, 'fldl2e': 16, 'fldlg2': 16, 'cvttps2pi': 17, 'cvtdq2ps': 17, 'cvtps2pd': 17, 'cvtdq2pd': 17, 'vcvtdq2pd': 17, 'cvtpd2ps': 17, 'cvtps2dq': 17, 'cvtps2pi': 17, 'cvtpi2ps': 17, 'cvttpd2dq': 17, 'cvttps2dq': 17, 'cvtpi2pd': 17, 'vcvtdq2ps': 17, 'vcvtpd2dq': 17, 'cvtpd2dq': 17, 'cmovew': 18, 'cmovnew': 18, 'cmovaew': 18, 'cmovgew': 18, 'cmovbew': 18, 'pcmpgtb': 19, 'pcmpgtd': 19, 'pcmpgtw': 19, 'pfcmpgt': 19, 'vpcmpgtb': 19, 'vfmadd213sd': 20, 'vfmadd231sd': 20, 'vfnmadd132sd': 20, 'vfnmadd231sd': 20, 'vfmadd213ss': 20, 'cvtsi2sd': 21, 'cvtss2sd': 21, 'cvtsi2ss': 21, 'cvttsd2si': 21, 'cvtsd2ss': 21, 'cvttss2si': 21, 'cvtsi2sdl': 21, 'cvtsd2si': 21, 'cvtsi2ssl': 21, 'cvtsi2sdq': 21, 'cvtsi2ssq': 21, 'cvtss2si': 21, 'vcvtsd2ss': 21, 'vcvtsi2ssl': 21, 'vcvtss2sd': 21, 'sqrtps': 22, 'rsqrtps': 22, 'sqrtsd': 22, 'str': 22, 'strd': 22, 'vrsqrtps': 22, 'pfrsqrt': 22, 'rsqrtss': 22, 'sqrtss': 22, 'pxor': 23, 'vpor': 23, 'vxorps': 23, 'vpxor': 23, 'vxorpd': 23, 'cmpxchgl': 24, 'cmpxchgq': 24, 'cmpxchgb': 24, 'cmpxchg8b': 24, 'cmpxchg16b': 24, 'itett': 25, 'ittee': 25, 'itte': 25, 'iteet': 25, 'itt': 25, 'ite': 25, 'iteee': 25, 'it': 25, 'itete': 25, 'ittte': 25, 'ittt': 25, 'itee': 25, 'ittet': 25, 'itet': 25, 'mcr2': 26, 'mcr': 26, 'mrrc2': 26, 'mrc': 26, 'mrc2': 26, 'mcrr': 26, 'mcrr2': 26, 'vldr': 27, 'ldrvs': 27, 'vpsrld': 27, 'vpsrlvd': 27, 'mulsd': 28, 'mulps': 28, 'mulss': 28, 'muls': 28, 'vmulss': 28, 'rorb': 29, 'rorl': 29, 'rorw': 29, 'rorq': 29, 'rors': 29, 'orr': 29, 'orrs': 29, 'popl': 30, 'popq': 30, 'popw': 30, 'vpop': 30, 'pop': 30, 'pshufd': 31, 'shufpd': 31, 'vshufps': 31, 'vshufpd': 31, 'vpshufb': 31, 'vpshufd': 31, 'ucomiss': 32, 'vcomisd': 32, 'comiss': 32, 'vcomiss': 32}


def get_top_k_keys(
        k,
        size=1000
):
    arr = os.listdir("/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/")
    if '.DS_Store' in arr:
        arr.remove('.DS_Store')

    clean = list(
        filter(
            lambda x: 'clean' in x,
            arr
        )
    )[:size]
    infected = list(
        filter(
            lambda x: 'infect' in x,
            arr
        )
    )[:size]

    total = 2 * size

    arr = clean + infected

    # TODO - get malicious and bengin files - this only looks at good files

    sets = [
        {
            'data': clean,
            'dict': {},
            'keys': []
        },
        {
            'data': infected,
            'dict': {},
            'keys': []
        }
    ]
    for s in sets:
        for i, file_name in enumerate(s['data']):
            with open(
                    f"/Volumes/MALWARE/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples/{file_name}") as file:
                try:
                    file_data = str(file.read()).split()
                except:
                    print(file_name)

            reduce_op_code_list_to_counts(file_data, s['dict'])

        s['dict'] = {k: v for k, v in sorted(s['dict'].items(), key=lambda item: item[1], reverse=True)}

        s['keys'] = list(
            filter(
                lambda z: not ('(' in z or '<' in z or '%' in z or '.' in z or '_' in z or '-' in z),
                list(s['dict'].copy().keys())
            )
        )#[:k]

        temp = s['dict'].copy().keys()

        for z in temp:
            if z not in s['keys']:
                s['dict'].pop(z)

    return sets[0], sets[1]


def get_union(
        k,
        verbose=False,
        clean=None,
        infected=None
):
    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    union = list(set(clean + infected))

    if verbose:
        print(f'Union - {k}')
        print(sorted(union))
        print(len(union))

    return union


def get_intersection(
        k,
        verbose=False,
        clean=None,
        infected=None
):
    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    intersection = list(filter(lambda x: x in clean, infected))

    if verbose:
        print(f'Intersection - {k}')
        print(sorted(intersection))
        print(len(intersection))

    return intersection


def get_disjoint(
        k,
        verbose=False,
        clean=None,
        infected=None
):
    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    disjoint = list(filter(lambda x: x not in clean, infected)) + list(filter(lambda x: x not in infected, clean))

    if verbose:
        print(f'Disjoint - {k}')
        print(sorted(list(filter(lambda x: x not in clean, infected))))
        print(sorted(list(filter(lambda x: x not in infected, clean))))
        print(sorted(disjoint))
        print(len(disjoint))

    return disjoint


def get_ratio(
        k,
        verbose=False,
        clean=None,
        infected=None,
        log_base=10,
        alpha=0.5
):
    def a(x):
        return max(x, .001)

    if clean is None or infected is None:
        clean, infected = get_top_k_keys(k)
    elif (clean is None or infected is None) and k is None:
        raise Exception("Either Clean and Infected or k must not be None")

    clean = clean['dict']
    infected = infected['dict']
    _set = list(set(list(clean.keys()) + list(infected.keys())))
    def combine(key):
        i = (0 if key not in infected else infected[key])
        c = (0 if key not in clean else clean[key])
        return i + c
    def div_log(key):
        i = (0 if key not in infected else infected[key]) + 0.001
        c = (0 if key not in clean else clean[key]) + 0.001
        return abs(math.log(i / c, log_base))
    combined_counts = np.array([combine(x) for x in _set]).astype(float)
    log_ratio = np.array([div_log(x) for x in _set]).astype(float)

    combined_counts *= 1 / max(combined_counts)
    log_ratio *= 1 / max(log_ratio)

    ratio_dict = {x: (alpha * combined_counts[i]) + ((1 - alpha) * log_ratio[i]) for i, x in enumerate(_set)}
    ratio_dict = {l: v for l, v in sorted(ratio_dict.items(), reverse=True, key=lambda item: item[1])}
    # print(ratio_dict)
    keys = list(ratio_dict.keys())[:k]

    if verbose:
        print(f'Ratio - {k}')
        print(sorted(keys))
        print(len(keys))

    return keys


if __name__ == "__main__":

    ks = [50, 100]
    clean, infected = get_top_k_keys(max(ks), 500)
    print('ratio, log 10, a - 0.1')
    get_ratio(50, True, clean, infected, alpha=0.1)
    print('ratio, log 10, a - 0.25')
    get_ratio(50, True, clean, infected, alpha=0.25)
    print('ratio, log 10, a - 0.5')
    get_ratio(50, True, clean, infected)
    print('ratio, log 10, a - 0.75')
    get_ratio(50, True, clean, infected, alpha=0.75)
    print('ratio, log 10, a - 0.9')
    get_ratio(50, True, clean, infected, alpha=0.9)
    raise Exception('')
    for k in ks:

        _clean = clean[:k]
        _infected = infected[:k]

        print(f"Clean only - {k}")
        print(sorted(_clean))
        print()
        print(f"Infected only - {k}")
        print(sorted(_infected))
        print()

        get_union(k, verbose=True, clean=_clean, infected=_infected)
        print()
        get_intersection(k, verbose=True, clean=_clean, infected=_infected)
        print()
        get_disjoint(k, verbose=True, clean=_clean, infected=_infected)
        print()

