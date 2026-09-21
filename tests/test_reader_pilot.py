from wrag.eval.reader_pilot import cited_answer, eligible


def test_route_ignores_benchmark_labels_and_preserves_counts():
    d = {'planos_compilados':[{'consulta':{'aggregation':'set'}}]}
    assert eligible('Who supports Alex?',d)
    assert not eligible('How many times has Alex visited?',d)
    assert not eligible('When did Alex leave?',{})


def test_item_failure_does_not_discard_other_supported_members():
    sources={'p1':'Alex plays the flute.', 'p2':'Alex plays the guitar.'}
    data={'items':[
        {'answer':'flute','evidence':[{'pid':'p1','quote':'Alex plays the flute.'}]},
        {'answer':'violin','evidence':[{'pid':'p2','quote':'Alex plays the violin.'}]},
        {'answer':'guitar','evidence':[{'pid':'p2','quote':'Alex plays the guitar.'}]}]}
    answer,rejected=cited_answer(data,sources)
    assert answer=='flute, guitar' and len(rejected)==1


def test_citation_must_be_in_delivered_passage_and_schema_must_be_valid():
    assert cited_answer(None,{})[0]=='insufficient information'
    assert cited_answer({'items':[{'answer':'x','evidence':[{'pid':'other','quote':'long enough'}]}]}, {})[0]=='insufficient information'


def test_paraphrase_answer_is_not_required_to_be_literal_quote():
    # Local validation establishes provenance only, not semantic entailment.
    answer,_=cited_answer({'items':[{'answer':'bowl','evidence':[{'pid':'p','quote':'I made two bowls.'}]}]}, {'p':'I made two bowls.'})
    assert answer=='bowl'
