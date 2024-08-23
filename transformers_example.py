from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
 
model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
 
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
# )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
 
article = '''
 
== BEGIN ARTICLE ==
 
Llama 2 : Open Foundation and Fine-Tuned Chat Models
Hugo Touvron∗Louis Martin†Kevin Stone†
Peter Albert Amjad Almahairi Yasmine Babaei Nikolay Bashlykov Soumya Batra
Prajjwal Bhargava Shruti Bhosale Dan Bikel Lukas Blecher Cristian Canton Ferrer Moya Chen
Guillem Cucurull David Esiobu Jude Fernandes Jeremy Fu Wenyin Fu Brian Fuller
Cynthia Gao Vedanuj Goswami Naman Goyal Anthony Hartshorn Saghar Hosseini Rui Hou
Hakan Inan Marcin Kardas Viktor Kerkez Madian Khabsa Isabel Kloumann Artem Korenev
Punit Singh Koura Marie-Anne Lachaux Thibaut Lavril Jenya Lee Diana Liskovich
Yinghai Lu Yuning Mao Xavier Martinet Todor Mihaylov Pushkar Mishra
Igor Molybog Yixin Nie Andrew Poulton Jeremy Reizenstein Rashi Rungta Kalyan Saladi
Alan Schelten Ruan Silva Eric Michael Smith Ranjan Subramanian Xiaoqing Ellen Tan Binh Tang
Ross Taylor Adina Williams Jian Xiang Kuan Puxin Xu Zheng Yan Iliyan Zarov Yuchen Zhang
Angela Fan Melanie Kambadur Sharan Narang Aurelien Rodriguez Robert Stojnic
Sergey Edunov Thomas Scialom∗
GenAI, Meta
Abstract
In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned
large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.
Our fine-tuned LLMs, called Llama 2-Chat , are optimized for dialogue use cases. Our
models outperform open-source chat models on most benchmarks we tested, and based on
ourhumanevaluationsforhelpfulnessandsafety,maybeasuitablesubstituteforclosed-
source models. We provide a detailed description of our approach to fine-tuning and safety
improvements of Llama 2-Chat in order to enable the community to build on our work and
contribute to the responsible development of LLMs.
∗Equal contribution, corresponding authors: {tscialom, htouvron}@meta.com
†Second author
2
Figure 1: Helpfulness human evaluation results for Llama
2-Chatcomparedtootheropen-sourceandclosed-source
models. Human raters compared model generations on ~4k
promptsconsistingofbothsingleandmulti-turnprompts.
The95%confidenceintervalsforthisevaluationarebetween
1%and2%. MoredetailsinSection3.4.2. Whilereviewing
these results, it is important to note that human evaluations
canbenoisyduetolimitationsofthepromptset,subjectivity
of the review guidelines, subjectivity of individual raters,
and the inherent difficulty of comparing generations.
Figure 2: Win-rate % for helpfulness and
safety between commercial-licensed base-
lines and Llama 2-Chat , according to GPT-
4. Tocomplementthehumanevaluation,we
used a more capable model, not subject to
ourownguidance. Greenareaindicatesour
modelisbetteraccordingtoGPT-4. Toremove
ties, we used win/ (win+loss). The orders in
whichthemodelresponsesarepresentedto
GPT-4arerandomlyswappedtoalleviatebias.
1 Introduction
Large Language Models (LLMs) have shown great promise as highly capable AI assistants that excel in
complex reasoning tasks requiring expert knowledge across a wide range of fields, including in specialized
domains such as programming and creative writing. They enable interaction with humans through intuitive
chat interfaces, which has led to rapid and widespread adoption among the general public.
ThecapabilitiesofLLMsareremarkableconsideringtheseeminglystraightforwardnatureofthetraining
methodology. Auto-regressivetransformersarepretrainedonanextensivecorpusofself-superviseddata,
followed by alignment with human preferences via techniques such as Reinforcement Learning with Human
Feedback(RLHF).Althoughthetrainingmethodologyissimple,highcomputationalrequirementshave
limited the development of LLMs to a few players. There have been public releases of pretrained LLMs
(such as BLOOM (Scao et al., 2022), LLaMa-1 (Touvron et al., 2023), and Falcon (Penedo et al., 2023)) that
match the performance of closed pretrained competitors like GPT-3 (Brown et al., 2020) and Chinchilla
(Hoffmann et al., 2022), but none of these models are suitable substitutes for closed “product” LLMs, such
asChatGPT,BARD,andClaude. TheseclosedproductLLMsareheavilyfine-tunedtoalignwithhuman
preferences, which greatly enhances their usability and safety. This step can require significant costs in
computeandhumanannotation,andisoftennottransparentoreasilyreproducible,limitingprogresswithin
the community to advance AI alignment research.
In this work, we develop and release Llama 2, a family of pretrained and fine-tuned LLMs, Llama 2 and
Llama 2-Chat , at scales up to 70B parameters. On the series of helpfulness and safety benchmarks we tested,
Llama 2-Chat models generally perform better than existing open-source models. They also appear to
be on par with some of the closed-source models, at least on the human evaluations we performed (see
Figures1and3). Wehavetakenmeasurestoincreasethesafetyofthesemodels,usingsafety-specificdata
annotation and tuning, as well as conducting red-teaming and employing iterative evaluations. Additionally,
thispapercontributesathoroughdescriptionofourfine-tuningmethodologyandapproachtoimproving
LLM safety. We hope that this openness will enable the community to reproduce fine-tuned LLMs and
continue to improve the safety of those models, paving the way for more responsible development of LLMs.
Wealsosharenovelobservationswemadeduringthedevelopmentof Llama 2 andLlama 2-Chat ,suchas
the emergence of tool usage and temporal organization of knowledge.
3
Figure 3: Safety human evaluation results for Llama 2-Chat compared to other open-source and closed-
source models. Human raters judged model generations for safety violations across ~2,000 adversarial
prompts consisting of both single and multi-turn prompts. More details can be found in Section 4.4. It is
importanttocaveatthesesafetyresultswiththeinherentbiasofLLMevaluationsduetolimitationsofthe
promptset,subjectivityofthereviewguidelines,andsubjectivityofindividualraters. Additionally,these
safety evaluations are performed using content standards that are likely to be biased towards the Llama
2-Chatmodels.
We are releasing the following models to the general public for research and commercial use‡:
1.Llama 2 ,anupdatedversionof Llama 1,trainedonanewmixofpubliclyavailabledata. Wealso
increasedthesizeofthepretrainingcorpusby40%,doubledthecontextlengthofthemodel,and
adoptedgrouped-queryattention(Ainslieetal.,2023). Wearereleasingvariantsof Llama 2 with
7B,13B,and70Bparameters. Wehavealsotrained34Bvariants,whichwereportoninthispaper
but are not releasing.§
2.Llama 2-Chat , a fine-tuned version of Llama 2 that is optimized for dialogue use cases. We release
variants of this model with 7B, 13B, and 70B parameters as well.
WebelievethattheopenreleaseofLLMs,whendonesafely,willbeanetbenefittosociety. LikeallLLMs,
Llama 2 is a new technology that carries potential risks with use (Bender et al., 2021b; Weidinger et al., 2021;
Solaimanet al.,2023). Testingconductedtodate hasbeeninEnglish andhasnot— andcouldnot— cover
all scenarios. Therefore, before deploying any applications of Llama 2-Chat , developers should perform
safetytestingand tuningtailoredtotheirspecificapplicationsofthemodel. Weprovidearesponsibleuse
guide¶and code examples‖to facilitate the safe deployment of Llama 2 andLlama 2-Chat . More details of
our responsible release strategy can be found in Section 5.3.
Theremainderofthispaperdescribesourpretrainingmethodology(Section2),fine-tuningmethodology
(Section 3), approach to model safety (Section 4), key observations and insights (Section 5), relevant related
work (Section 6), and conclusions (Section 7).
‡https://ai.meta.com/resources/models-and-libraries/llama/
§We are delaying the release of the 34B model due to a lack of time to sufficiently red team.
¶https://ai.meta.com/llama
‖https://github.com/facebookresearch/llama
4
Figure4: Trainingof Llama 2-Chat : Thisprocessbeginswiththe pretraining ofLlama 2 usingpublicly
availableonlinesources. Followingthis,wecreateaninitialversionof Llama 2-Chat throughtheapplication
ofsupervised fine-tuning . Subsequently, the model is iteratively refined using Reinforcement Learning
with Human Feedback (RLHF) methodologies, specifically through rejection sampling and Proximal Policy
Optimization(PPO).ThroughouttheRLHFstage,theaccumulationof iterativerewardmodelingdata in
parallel with model enhancements is crucial to ensure the reward models remain within distribution.
2 Pretraining
Tocreatethenewfamilyof Llama 2models,webeganwiththepretrainingapproachdescribedinTouvronetal.
(2023), using an optimized auto-regressive transformer, but made several changes to improve performance.
Specifically,weperformedmorerobustdatacleaning,updatedourdatamixes,trainedon40%moretotal
tokens,doubledthecontextlength,andusedgrouped-queryattention(GQA)toimproveinferencescalability
for our larger models. Table 1 compares the attributes of the new Llama 2 models with the Llama 1 models.
2.1 Pretraining Data
Our training corpus includes a new mix of data from publicly available sources, which does not include data
fromMeta’sproductsorservices. Wemadeanefforttoremovedatafromcertainsitesknowntocontaina
highvolumeofpersonalinformationaboutprivateindividuals. Wetrainedon2trilliontokensofdataasthis
providesagoodperformance–costtrade-off,up-samplingthemostfactualsourcesinanefforttoincrease
knowledge and dampen hallucinations.
Weperformedavarietyofpretrainingdatainvestigationssothatuserscanbetterunderstandthepotential
capabilities and limitations of our models; results can be found in Section 4.1.
2.2 Training Details
We adopt most of the pretraining setting and model architecture from Llama 1 . We use the standard
transformer architecture (Vaswani et al., 2017), apply pre-normalization using RMSNorm (Zhang and
Sennrich, 2019), use the SwiGLU activation function (Shazeer, 2020), and rotary positional embeddings
(RoPE, Su et al. 2022). The primary architectural differences from Llama 1 include increased context length
andgrouped-queryattention(GQA).WedetailinAppendixSectionA.2.1eachofthesedifferenceswith
ablation experiments to demonstrate their importance.
Hyperparameters. We trained using the AdamW optimizer (Loshchilov and Hutter, 2017), with β1=
0.9, β2= 0.95,eps= 10−5. We use a cosine learning rate schedule, with warmup of 2000 steps, and decay
finallearningratedownto10%ofthepeaklearningrate. Weuseaweightdecayof 0.1andgradientclipping
of1.0. Figure 5 (a) shows the training loss for Llama 2 with these hyperparameters.
5
Training Data Params Context
LengthGQA Tokens LR
Llama 1See Touvron et al.
(2023)7B 2k ✗ 1.0T 3.0×10−4
13B 2k ✗ 1.0T 3.0×10−4
33B 2k ✗ 1.4T 1.5×10−4
65B 2k ✗ 1.4T 1.5×10−4
Llama 2A new mix of publicly
available online data7B 4k ✗ 2.0T 3.0×10−4
13B 4k ✗ 2.0T 3.0×10−4
34B 4k ✓ 2.0T 1.5×10−4
70B 4k ✓ 2.0T 1.5×10−4
Table 1: Llama 2 family of models. Token counts refer to pretraining data only. All models are trained with
a global batch-size of 4M tokens. Bigger models — 34B and 70B — use Grouped-Query Attention (GQA) for
improved inference scalability.
0 250 500 750 1000 1250 1500 1750 2000
Processed Tokens (Billions)1.41.51.61.71.81.92.02.12.2Train PPLLlama-2
7B
13B
34B
70B
Figure 5: Training Loss for Llama 2 models. We compare the training loss of the Llama 2 family of models.
We observe that after pretraining on 2T Tokens, the models still did not show any sign of saturation.
Tokenizer. Weusethesametokenizeras Llama 1;itemploysabytepairencoding(BPE)algorithm(Sennrich
etal.,2016)usingtheimplementationfromSentencePiece(KudoandRichardson,2018). Aswith Llama 1,
we split all numbers into individual digits and use bytes to decompose unknown UTF-8 characters. The total
vocabulary size is 32k tokens.
2.2.1 Training Hardware & Carbon Footprint
TrainingHardware. WepretrainedourmodelsonMeta’sResearchSuperCluster(RSC)(LeeandSengupta,
2022)aswellasinternalproductionclusters. BothclustersuseNVIDIAA100s. Therearetwokeydifferences
between the two clusters, with the first being the type of interconnect available: RSC uses NVIDIA Quantum
InfiniBandwhileourproductionclusterisequippedwithaRoCE(RDMAoverconvergedEthernet)solution
based on commodity ethernet Switches. Both of these solutions interconnect 200 Gbps end-points. The
seconddifferenceistheper-GPUpowerconsumptioncap—RSCuses400Wwhileourproductioncluster
uses350W.Withthistwo-clustersetup,wewereabletocomparethesuitabilityofthesedifferenttypesof
interconnectforlargescaletraining. RoCE(whichisamoreaffordable,commercialinterconnectnetwork)
6
Time
(GPU hours)Power
Consumption (W)Carbon Emitted
(tCO 2eq)
Llama 27B 184320 400 31.22
13B 368640 400 62.44
34B 1038336 350 153.90
70B 1720320 400 291.42
Total 3311616 539.00
Table 2: CO2emissions during pretraining. Time: total GPU time required for training each model. Power
Consumption: peak power capacity per GPU device for the GPUs used adjusted for power usage efficiency.
100%oftheemissionsaredirectlyoffsetbyMeta’ssustainabilityprogram,andbecauseweareopenlyreleasing
these models, the pretraining costs do not need to be incurred by others.
can scale almost as well as expensive Infiniband up to 2000 GPUs, which makes pretraining even more
democratizable. On A100s with RoCE and GPU power capped at 350W, our optimized codebase reached up
to 90% of the performance of RSC using IB interconnect and 400W GPU power.
Carbon Footprint of Pretraining. Following preceding research (Bender et al., 2021a; Patterson et al., 2021;
Wu et al., 2022; Dodge et al., 2022) and using power consumption estimates of GPU devices and carbon
efficiency, we aim tocalculate thecarbon emissions resultingfrom the pretrainingof Llama 2 models. The
actualpowerusageofaGPUisdependentonitsutilizationandislikelytovaryfromtheThermalDesign
Power(TDP)thatweemployasanestimationforGPUpower. Itisimportanttonotethatourcalculations
do not account for further power demands, such as those from interconnect or non-GPU server power
consumption,norfromdatacentercoolingsystems. Additionally,thecarbonoutputrelatedtotheproduction
of AI hardware, like GPUs, could add to the overall carbon footprint as suggested by Gupta et al. (2022b,a).
Table 2 summarizes the carbon emission for pretraining the Llama 2 family of models. A cumulative of
3.3M GPUhours ofcomputation wasperformed onhardware oftype A100-80GB (TDPof 400Wor 350W).
We estimate the total emissions for training to be 539 tCO 2eq, of which 100% were directly offset by Meta’s
sustainability program.∗∗Our open release strategy also means that these pretraining costs will not need to
be incurred by other companies, saving more global resources.
2.3 Llama 2 Pretrained Model Evaluation
In this section, we report the results for the Llama 1 andLlama 2 base models, MosaicML Pretrained
Transformer(MPT)††models,andFalcon(Almazroueietal.,2023)modelsonstandardacademicbenchmarks.
For all the evaluations, we use our internal evaluations library. We reproduce results for the MPT and Falcon
modelsinternally. Forthesemodels,wealwayspickthebestscorebetweenourevaluationframeworkand
any publicly reported results.
InTable3,wesummarizetheoverallperformanceacrossasuiteofpopularbenchmarks. Notethatsafety
benchmarks are shared in Section 4.1. The benchmarks are grouped into the categories listed below. The
results for all the individual benchmarks are available in Section A.2.2.
•Code.Wereporttheaveragepass@1scoresofourmodelsonHumanEval(Chenetal.,2021)and
MBPP (Austin et al., 2021).
•CommonsenseReasoning. WereporttheaverageofPIQA(Bisketal.,2020),SIQA(Sapetal.,2019),
HellaSwag (Zellers et al., 2019a), WinoGrande (Sakaguchi et al., 2021), ARC easy and challenge
(Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018), and CommonsenseQA (Talmor et al.,
2018). We report 7-shot results for CommonSenseQA and 0-shot results for all other benchmarks.
•World Knowledge. We evaluate the 5-shot performance on NaturalQuestions (Kwiatkowski et al.,
2019) and TriviaQA (Joshi et al., 2017) and report the average.
•Reading Comprehension. For reading comprehension, we report the 0-shot average on SQuAD
(Rajpurkar et al., 2018), QuAC (Choi et al., 2018), and BoolQ (Clark et al., 2019).
∗∗https://sustainability.fb.com/2021-sustainability-report/
††https://www.mosaicml.com/blog/mpt-7b
7
Model Size CodeCommonsense
ReasoningWorld
KnowledgeReading
ComprehensionMath MMLU BBH AGI Eval
MPT7B 20.5 57.4 41.0 57.5 4.9 26.8 31.0 23.5
30B 28.9 64.9 50.0 64.7 9.1 46.9 38.0 33.8
Falcon7B 5.6 56.1 42.8 36.0 4.6 26.2 28.0 21.2
40B 15.2 69.2 56.7 65.7 12.6 55.4 37.1 37.0
Llama 17B 14.1 60.8 46.2 58.5 6.95 35.1 30.3 23.9
13B 18.9 66.1 52.6 62.3 10.9 46.9 37.0 33.9
33B 26.0 70.0 58.4 67.6 21.4 57.8 39.8 41.7
65B 30.7 70.7 60.5 68.6 30.8 63.4 43.5 47.6
Llama 27B 16.8 63.9 48.9 61.3 14.6 45.3 32.6 29.3
13B 24.5 66.9 55.4 65.8 28.7 54.8 39.4 39.1
34B 27.8 69.9 58.7 68.0 24.2 62.6 44.1 43.4
70B37.5 71.9 63.6 69.4 35.2 68.9 51.2 54.2
Table3: Overallperformanceongroupedacademicbenchmarkscomparedtoopen-sourcebasemodels.
•MATH. We report the average of the GSM8K (8 shot) (Cobbe et al., 2021) and MATH (4 shot)
(Hendrycks et al., 2021) benchmarks at top 1.
•Popular Aggregated Benchmarks . We report the overall results for MMLU (5 shot) (Hendrycks
et al., 2020), Big Bench Hard (BBH) (3 shot) (Suzgun et al., 2022), and AGI Eval (3–5 shot) (Zhong
et al., 2023). For AGI Eval, we only evaluate on the English tasks and report the average.
As shown in Table 3, Llama 2 models outperform Llama 1 models. In particular, Llama 2 70B improves the
resultsonMMLUandBBHby ≈5and≈8points,respectively,comparedto Llama 1 65B.Llama 2 7Band30B
modelsoutperformMPTmodelsofthecorrespondingsizeonallcategoriesbesidescodebenchmarks. Forthe
Falcon models, Llama 2 7B and 34B outperform Falcon 7B and 40B models on all categories of benchmarks.
Additionally, Llama 2 70B model outperforms all open-source models.
In addition to open-source models, we also compare Llama 2 70B results to closed-source models. As shown
in Table 4, Llama 2 70B is close to GPT-3.5 (OpenAI, 2023) on MMLU and GSM8K, but there is a significant
gaponcodingbenchmarks. Llama 2 70BresultsareonparorbetterthanPaLM(540B)(Chowdheryetal.,
2022)onalmostallbenchmarks. Thereisstillalargegapinperformancebetween Llama 2 70BandGPT-4
and PaLM-2-L.
We also analysed the potential data contamination and share the details in Section A.6.
Benchmark (shots) GPT-3.5 GPT-4 PaLM PaLM-2-L Llama 2
MMLU (5-shot) 70.0 86.4 69.3 78.3 68.9
TriviaQA (1-shot) – – 81.4 86.1 85.0
Natural Questions (1-shot) – – 29.3 37.5 33.0
GSM8K (8-shot) 57.1 92.0 56.5 80.7 56.8
HumanEval (0-shot) 48.1 67.0 26.2 – 29.9
BIG-Bench Hard (3-shot) – – 52.3 65.7 51.2
Table 4: Comparison to closed-source models on academic benchmarks. Results for GPT-3.5 and GPT-4
are from OpenAI (2023). Results for the PaLM model are from Chowdhery et al. (2022). Results for the
PaLM-2-L are from Anil et al. (2023).
 
== END ARTICLE ==
 
'''
 
chat = [
    {
        "role": "system",
        "content": "You are Hermes 2, an unbiased AI assistant, and you always answer to the best of your ability."
    },
    {
        "role": "user",
        "content": (
            "You are given a partial and unparsed scientific article, please read it carefully and complete the "
            f"request below.{article}Please summarize the article in 5 sentences."
        )
    },
]
processed_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(processed_chat, return_tensors='pt').to(model.device)
 
streamer = TextStreamer(tokenizer)
 
# Run and measure generate
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
torch.cuda.reset_peak_memory_stats(model.device)
torch.cuda.empty_cache()
torch.cuda.synchronize()
start_event.record()
# generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512, streamer=streamer)
generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512, streamer=streamer, prompt_lookup_num_tokens=10)
end_event.record()
torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated(model.device)
print("Max memory (MB): ", max_memory * 1e-6)
new_tokens = generation_output.shape[1] - input_ids.input_ids.shape[1]
print("Throughput (tokens/sec): ", new_tokens / (start_event.elapsed_time(end_event) * 1.0e-3))
