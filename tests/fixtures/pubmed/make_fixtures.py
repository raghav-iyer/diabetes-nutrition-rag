"""Generate synthetic PubMed XML fixtures (fake PMIDs 900000xx). Not real records."""
from xml.sax.saxutils import escape
import sys, pathlib

FILLER = ("Participants were followed with standardised dietary assessment and outcomes were "
          "ascertained by trained staff blinded to allocation where applicable. ")

def art(pmid, title, abstract, year=None, medline_date=None, article_date=None, pub_types=("Journal Article",),
        mesh=("Humans",), status="MEDLINE", doi=None, eloc_doi=None, pmc=None, langs=("eng",),
        corrections=(), coi=None, grants=(), structured=False, major=(), nlm="0001001"):
    if structured:
        ab = "".join(f'<AbstractText Label="{escape(l)}">{escape(t)}</AbstractText>' for l, t in abstract)
    else:
        ab = f"<AbstractText>{escape(abstract)}</AbstractText>" if abstract else ""
    pd = ""
    if year: pd = f"<Year>{year}</Year><Month>Mar</Month>"
    elif medline_date: pd = f"<MedlineDate>{medline_date}</MedlineDate>"
    ad = f'<ArticleDate DateType="Electronic"><Year>{article_date}</Year><Month>01</Month><Day>05</Day></ArticleDate>' if article_date else ""
    pts = "".join(f"<PublicationType>{escape(p)}</PublicationType>" for p in pub_types)
    mh = "".join(
        f'<MeshHeading><DescriptorName MajorTopicYN="{"Y" if m in major else "N"}">{escape(m)}</DescriptorName></MeshHeading>'
        for m in mesh)
    mhl = f"<MeshHeadingList>{mh}</MeshHeadingList>" if mesh else ""
    cc = "".join(f'<CommentsCorrections RefType="{r}"><RefSource>x</RefSource></CommentsCorrections>' for r in corrections)
    ccl = f"<CommentsCorrectionsList>{cc}</CommentsCorrectionsList>" if corrections else ""
    el = f'<ELocationID EIdType="doi" ValidYN="Y">{eloc_doi}</ELocationID>' if eloc_doi else ""
    gl = ("<GrantList>" + "".join(f"<Grant><Agency>{escape(g)}</Agency><Country>India</Country></Grant>" for g in grants)
          + "</GrantList>") if grants else ""
    ids = f'<ArticleId IdType="pubmed">{pmid}</ArticleId>'
    if doi: ids += f'<ArticleId IdType="doi">{doi}</ArticleId>'
    if pmc: ids += f'<ArticleId IdType="pmc">{pmc}</ArticleId>'
    langs_x = "".join(f"<Language>{l}</Language>" for l in langs)
    coi_x = f"<CoiStatement>{escape(coi)}</CoiStatement>" if coi else ""
    return f"""<PubmedArticle><MedlineCitation Status="{status}" Owner="NLM"><PMID Version="1">{pmid}</PMID>
<Article PubModel="Print"><Journal><ISSN IssnType="Electronic">1234-5678</ISSN><JournalIssue><PubDate>{pd}</PubDate></JournalIssue><Title>Journal of Test Nutrition</Title></Journal>
<ArticleTitle>{escape(title)}</ArticleTitle>{el}<Abstract>{ab}</Abstract>{langs_x}{gl}<PublicationTypeList>{pts}</PublicationTypeList>{ad}</Article>
<MedlineJournalInfo><Country>England</Country><MedlineTA>J Test Nutr</MedlineTA><NlmUniqueID>{nlm}</NlmUniqueID></MedlineJournalInfo>{ccl}{mhl}{coi_x}</MedlineCitation><PubmedData><ArticleIdList>{ids}</ArticleIdList></PubmedData></PubmedArticle>
"""

def wrap(arts):
    return '<?xml version="1.0" ?>\n<!DOCTYPE PubmedArticleSet>\n<PubmedArticleSet>\n' + "".join(arts) + "</PubmedArticleSet>\n"

base = [
 art(90000001, "Low glycaemic index diet versus usual care for preventing type 2 diabetes in Indian adults with prediabetes: a randomised trial",
     [("BACKGROUND", "South Asian adults have high diabetes risk."), ("METHODS", "We randomised n = 240 adults with prediabetes to a low glycaemic index diet or usual care. " + FILLER),
      ("RESULTS", "Diabetes incidence was lower with the intervention."), ("CONCLUSIONS", "A low GI diet reduced progression.")],
     structured=True, year=2020, pub_types=("Journal Article", "Randomized Controlled Trial"),
     mesh=("Humans", "Prediabetic State", "Diet"), major=("Prediabetic State",), doi="10.1000/TEST.0001", pmc="PMC9000001",
     coi="The authors declare no competing interests.", grants=("Indian Council of Medical Research",)),
 art(90000002, "Mediterranean diet and gestational diabetes: a meta-analysis",
     "We pooled 23 trials including 4,500 pregnant women. Mediterranean-style diets reduced gestational diabetes risk compared with usual care. " + FILLER,
     year=2018, pub_types=("Journal Article", "Meta-Analysis", "Review"), mesh=("Humans", "Female", "Pregnancy", "Diabetes, Gestational"), doi="10.1000/test.0002", pmc="PMC9000002"),
 art(90000003, "Dietary pulses and weight loss: a systematic review",
     "This systematic review summarises controlled feeding studies of pulses and body weight in adults with overweight. " + FILLER,
     medline_date="2015 Jan-Feb", status="PubMed-not-MEDLINE", mesh=("Humans", "Weight Loss")),
 art(90000004, "Nutrition in pregnancy: an overview for clinicians",
     "This overview discusses energy, protein, iron, folate and iodine requirements during pregnancy and lactation, drawing on practice experience. " + FILLER,
     year=2016, pub_types=("Journal Article", "Review"), mesh=("Humans", "Female", "Pregnancy")),
 art(90000005, "Dietary patterns and incident diabetes: review of a long-running population study",
     "We review findings on dietary patterns and incident type 2 diabetes from repeated diet questionnaires in adults followed for twenty years. " + FILLER,
     year=2019, pub_types=("Journal Article", "Review"), mesh=("Humans", "Cohort Studies", "Diabetes Mellitus, Type 2")),
 art(90000006, "Why calorie counting fails",
     "In this editorial we argue that calorie counting is an unhelpful frame for weight management and that clinicians should focus on food quality rather than energy arithmetic. " + FILLER,
     year=2021, pub_types=("Editorial",), mesh=("Humans", "Weight Loss")),
 art(90000007, "Retracted: vitamin D supplements prevent diabetes in pregnancy",
     "Women were randomised to vitamin D or placebo. Fewer cases of gestational diabetes occurred with vitamin D in this randomized controlled trial. " + FILLER,
     year=2017, pub_types=("Journal Article", "Randomized Controlled Trial", "Retracted Publication"), mesh=("Humans", "Female")),
 art(90000008, "Intermittent fasting and weight maintenance",
     "Adults were randomized in a parallel-group trial of intermittent fasting versus daily restriction with weight maintenance as the outcome. " + FILLER,
     year=2019, pub_types=("Journal Article",), mesh=("Humans",), corrections=("RetractionIn",)),
 art(90000009, "High protein diet for weight loss: a randomised controlled trial",
     "We enrolled 120 adults with obesity in a randomized controlled trial of high protein versus standard protein diets over 12 months. " + FILLER,
     year=2022, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=("Humans", "Obesity"), corrections=("ExpressionOfConcernIn", "ErratumIn")),
 art(90000010, "Ernaehrung in der Schwangerschaft",
     "Diese Studie untersucht die Ernaehrung schwangerer Frauen in einer randomisierten Studie mit 300 Teilnehmerinnen ueber zwoelf Monate. " + FILLER,
     year=2014, pub_types=("Journal Article",), mesh=("Humans", "Female"), langs=("ger",)),
 art(90000011, "Short note on breakfast", "Breakfast was associated with weight.",
     year=2015, mesh=("Humans",)),
 art(90000012, "High fat feeding in pregnant rats",
     "Pregnant rats were fed a high fat diet and offspring glucose tolerance was measured at weaning in a controlled laboratory experiment. " + FILLER,
     year=2016, mesh=("Animals", "Rats", "Female", "Pregnancy")),
 art(90000013, "Millet-based diets and glycaemia in Indian women with gestational diabetes",
     "A cross-sectional survey of 85 women with gestational diabetes in Chennai assessed millet intake and fasting glucose. " + FILLER,
     article_date=2026, status="In-Process", mesh=(), eloc_doi="10.1000/TEST.0013"),
 art(90000014, "Low glycaemic index diet versus usual care in prediabetes (duplicate record)",
     "We randomised n = 240 adults with prediabetes to a low glycaemic index diet or usual care in a randomised trial. " + FILLER,
     year=2020, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=("Humans",), doi="10.1000/test.0001"),
 art(90000015, "Maternal diet quality and birth weight in a multicentre cohort",
     "This prospective cohort of 1,950 pregnant women examined diet quality and birth weight across five centres. " + FILLER,
     year=2013, pub_types=("Journal Article", "Observational Study", "Multicenter Study"), mesh=("Humans", "Cohort Studies", "Pregnancy")),
 art(90000016, "Retraction notice: intermittent fasting and weight maintenance",
     "This article has been retracted at the request of the editors after concerns were raised about the integrity of the data reported in the trial of intermittent fasting. " + FILLER,
     year=2023, pub_types=("Retraction of Publication",), mesh=("Humans",)),
 art(90000017, "Sugar-sweetened beverages and weight in adolescents",
     "A cross-sectional study (n = 35) of adolescents examined sugar-sweetened beverage intake and body mass index in urban schools. " + FILLER,
     year=2011, mesh=("Humans", "Adolescent")),

 # --- Phase 1 additions ---
 art(90000019, "Diet quality in pregnancy and offspring adiposity: a prospective cohort study",
     "In this prospective cohort, 5,000 pregnant women completed food frequency questionnaires and offspring adiposity was measured at age five. " + FILLER,
     year=2019, mesh=("Humans", "Female", "Pregnancy", "Prospective Studies")),
 art(90000020, "Breakfast skipping and weight gain: a small prospective cohort",
     "This prospective cohort followed 300 adults for weight change according to breakfast habits over four years of follow-up. " + FILLER,
     year=2018, mesh=("Humans", "Prospective Studies")),
 art(90000021, "Protocol for a randomised trial of millet diets in prediabetes",
     "We describe the protocol of a randomized controlled trial that will enrol 600 adults with prediabetes in southern India over two years. " + FILLER,
     year=2024, pub_types=("Journal Article", "Clinical Trial Protocol", "Randomized Controlled Trial"), mesh=("Humans", "Prediabetic State")),
 art(90000022, "Probiotic yoghurt and gestational diabetes: a randomised controlled trial",
     "We randomized 400 pregnant women to probiotic yoghurt or control and measured gestational diabetes incidence at 28 weeks of gestation. " + FILLER,
     year=2020, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=("Humans", "Female", "Pregnancy"), doi="10.1000/test.0022"),
 art(90000023, "Very low calorie diet for weight loss: randomised trial in a non-indexed journal",
     "We randomized 150 adults with obesity to a very low calorie diet or usual care and measured weight at one year of follow-up. " + FILLER,
     year=2022, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=(), status="Publisher", nlm="9999999"),
 art(90000024, "Pulse-based meals and HbA1c in prediabetes: a randomised controlled trial",
     "We randomized 220 adults with prediabetes to pulse-based meals or usual diet; HbA1c was measured at six months of follow-up. " + FILLER,
     year=2026, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=(), status="In-Process"),
 art(90000025, "Dietary habits of urban Indian adults",
     "Previous meta-analyses have suggested that dietary patterns influence diabetes risk. We surveyed 900 adults about their usual diet in a cross-sectional design. " + FILLER,
     year=2021, mesh=("Humans", "India")),
 art(90000026, "Green tea extract for weight maintenance: a randomised controlled trial",
     "We randomized 180 adults who had lost weight to green tea extract or placebo and measured weight regain over twelve months. " + FILLER,
     year=2017, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=("Humans",), doi="10.1000/test.0026"),
]

refresh = [
 # 90000001 re-fetched, now carrying a retraction link.
 base[0].replace('<MeshHeadingList>', '<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><RefSource>x</RefSource></CommentsCorrections></CommentsCorrectionsList><MeshHeadingList>'),
 art(90000018, "Lactation and postpartum weight retention: a randomized controlled trial of dietary counselling",
     "We randomized 410 lactating women to dietary counselling or standard care and measured postpartum weight retention at 12 months. " + FILLER,
     year=2024, pub_types=("Journal Article", "Randomized Controlled Trial"), mesh=("Humans", "Female", "Lactation")),
]

out = pathlib.Path(sys.argv[1])
(out / "pubmed_base.xml").write_text(wrap(base))
(out / "pubmed_refresh.xml").write_text(wrap(refresh))
print("ok", len(base), len(refresh))
