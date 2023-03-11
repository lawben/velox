/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <filesystem>
#include <iostream>

#include <folly/init/Init.h>
#include "velox/connectors/Connector.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FieldReference.h"
#include "velox/tpch/gen/TpchGen.h"

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/common/SelectiveIntegerColumnReader.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Split.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::connector::tpch;

namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}

void ensureTaskCompletion(exec::Task* task) {
  // ASSERT_TRUE requires a function with return type void.
  ASSERT_TRUE(waitForTaskCompletion(task));
}

void printResults(const std::vector<RowVectorPtr>& results) {
  std::cout << "Results:" << std::endl;
  bool printType = true;
  for (const auto& vector : results) {
    // Print RowType only once.
    if (printType) {
      std::cout << vector->type()->asRow().toString() << std::endl;
      printType = false;
    }
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      std::cout << vector->toString(i) << std::endl;
    }
  }
}

constexpr size_t kRowsPerSplit = 10'000;

std::unordered_map<tpch::Table, std::vector<RowVectorPtr>> tableSplits;

RowVectorPtr getTpchData(
    tpch::Table table,
    size_t maxRows,
    size_t offset,
    double scaleFactor,
    memory::MemoryPool* pool) {
  switch (table) {
    case tpch::Table::TBL_PART:
      return tpch::genTpchPart(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_SUPPLIER:
      return tpch::genTpchSupplier(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_PARTSUPP:
      return tpch::genTpchPartSupp(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_CUSTOMER:
      return tpch::genTpchCustomer(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_ORDERS:
      return tpch::genTpchOrders(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_LINEITEM:
      return tpch::genTpchLineItem(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_NATION:
      return tpch::genTpchNation(maxRows, offset, scaleFactor, pool);
    case tpch::Table::TBL_REGION:
      return tpch::genTpchRegion(maxRows, offset, scaleFactor, pool);
  }
}

void generateTable(tpch::Table table, uint32_t scaleFactor) {
  auto pool = memory::getDefaultMemoryPool();
  const std::string table_name = std::string{tpch::toTableName(table)};

  std::vector<RowVectorPtr>& splits = tableSplits[table];

  const uint64_t numRows = tpch::getRowCount(table, scaleFactor);
  size_t offset = 0;
  uint64_t rowCount = 0;

  while (rowCount < numRows) {
    auto data =
        getTpchData(table, kRowsPerSplit, offset, scaleFactor, pool.get());

    // Resize is for lineitems table since the rowCount can exceed the numRows.
    data->resize(
        std::min((numRows - rowCount), static_cast<uint64_t>(data->size())));
    offset += kRowsPerSplit;
    rowCount += data->size();
    splits.emplace_back(std::move(data));
  }
}

void generateTpchData(uint32_t scaleFactor) {
  std::cout << "Generating TPCH data with SF=" << scaleFactor << std::endl;
  auto start = std::chrono::steady_clock::now();
  generateTable(tpch::Table::TBL_LINEITEM, scaleFactor);
  generateTable(tpch::Table::TBL_ORDERS, scaleFactor);
  generateTable(tpch::Table::TBL_PART, scaleFactor);
  generateTable(tpch::Table::TBL_SUPPLIER, scaleFactor);
  generateTable(tpch::Table::TBL_PARTSUPP, scaleFactor);
  generateTable(tpch::Table::TBL_CUSTOMER, scaleFactor);
  generateTable(tpch::Table::TBL_NATION, scaleFactor);
  generateTable(tpch::Table::TBL_REGION, scaleFactor);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<float>(end - start).count();
  std::cout << "Generating completed in " << duration << " s." << std::endl;
}

class VectorFormatData : public dwio::common::FormatData {
 public:
  VectorFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const std::vector<thrift::RowGroup>& rowGroups,
      memory::MemoryPool& pool)
      : {}

  void readNulls(
      vector_size_t numValues,
      const uint64_t* incomingNulls,
      BufferPtr& nulls,
      bool nullsOnly) override {}

  uint64_t skipNulls(uint64_t numValues, bool nullsOnly) override {
    return 0;
  }

  uint64_t skip(uint64_t numValues) override {
    return 0;
  }

  bool hasNulls() const override {
    return false;
  }

  PositionProvider seekToRowGroup(uint32_t index) override {
    return dwio::common::PositionProvider(__1::vector());
  }

  void filterRowGroups(
      const common::ScanSpec& scanSpec,
      uint64_t rowsPerRowGroup,
      const StatsContext& writerContext,
      FilterRowGroupsResult& result) override {}

  bool parentNullsInLeaves() const override {
    return FormatData::parentNullsInLeaves();
  }

  template <typename Visitor>
  void readWithVisitor(Visitor visitor) {
    reader_->readWithVisitor(visitor);
  }

 protected:
  memory::MemoryPool& pool_;
  std::shared_ptr<const ParquetTypeWithId> type_;
  const std::vector<thrift::RowGroup>& rowGroups_;
  // Streams for this column in each of 'rowGroups_'. Will be created on or
  // ahead of first use, not at construction.
  std::vector<std::unique_ptr<dwio::common::SeekableInputStream>> streams_;

  const uint32_t maxDefine_;
  const uint32_t maxRepeat_;
  int64_t rowsInRowGroup_;
  std::unique_ptr<PageReader> reader_;

  // Nulls derived from leaf repdefs for non-leaf readers.
  BufferPtr presetNulls_;

  // Number of valid bits in 'presetNulls_'.
  int32_t presetNullsSize_{0};

  // Count of leading skipped positions in 'presetNulls_'
  int32_t presetNullsConsumed_{0};
};

class VectorFormatParams : public dwio::common::FormatParams {
 public:
  VectorFormatParams(memory::MemoryPool& pool)
      : dwio::common::FormatParams{pool} {}

  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const common::ScanSpec& scanSpec) override {
    return std::make_unique<VectorFormatData>(type, , pool());
  }
};

class VectorIntegerReader : public dwio::common::SelectiveIntegerColumnReader {
 public:
  VectorIntegerReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      VectorFormatParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveIntegerColumnReader(
            std::move(requestedType),
            params,
            scanSpec,
            dataType->type) {}

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override {
    VELOX_WIDTH_DISPATCH(
        type_->cppSizeInBytes(), prepareRead, offset, rows, nullptr);
    readCommon<VectorIntegerReader>(rows);
  }

  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor) {
    formatData_->as<VectorFormatData>().readWithVisitor(visitor);
    readOffset_ += rows.back() + 1;
  }
};

class TpchBenchmarkSource : public connector::DataSource {
 public:
  TpchBenchmarkSource(
      const std::shared_ptr<const RowType>& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      memory::MemoryPool* FOLLY_NONNULL pool,
      connector::ExpressionEvaluator* expressionEvaluator,
      memory::MemoryAllocator* allocator)
      : pool_(pool),
        expressionEvaluator_{expressionEvaluator},
        allocator_{allocator} {
    auto tpchTableHandle =
        std::dynamic_pointer_cast<TpchBenchmarkTableHandle>(tableHandle);
    VELOX_CHECK_NOT_NULL(
        tpchTableHandle,
        "TableHandle must be an instance of TpchBenchmarkTableHandle");
    tpchTable_ = tpchTableHandle->getTable();

    dataSplits_ = &tableSplits.at(tpchTable_);

    auto tpchTableSchema = getTableSchema(tpchTableHandle->getTable());
    VELOX_CHECK_NOT_NULL(tpchTableSchema, "TpchSchema can't be null.");

    outputColumnMappings_.reserve(outputType->size());

    for (const auto& outputName : outputType->names()) {
      auto it = columnHandles.find(outputName);
      VELOX_CHECK(
          it != columnHandles.end(),
          "ColumnHandle is missing for output column '{}' on table '{}'",
          outputName,
          toTableName(tpchTable_));

      auto handle = std::dynamic_pointer_cast<TpchColumnHandle>(it->second);
      VELOX_CHECK_NOT_NULL(
          handle,
          "ColumnHandle must be an instance of TpchColumnHandle "
          "for '{}' on table '{}'",
          handle->name(),
          toTableName(tpchTable_));

      auto idx = tpchTableSchema->getChildIdxIfExists(handle->name());
      VELOX_CHECK(
          idx != std::nullopt,
          "Column '{}' not found on TPC-H table '{}'.",
          handle->name(),
          toTableName(tpchTable_));
      outputColumnMappings_.emplace_back(*idx);
    }
    outputType_ = outputType;

    /// BEGIN Copied from HiveDataSource
    const auto& remainingFilter = tpchTableHandle->remainingFilter();
    if (remainingFilter) {
      metadataFilter_ = std::make_shared<common::MetadataFilter>(
          *scanSpec_, *remainingFilter);
      remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);

      // Remaining filter may reference columns that are not used otherwise,
      // e.g. are not being projected out and are not used in range filters.
      // Make sure to add these columns to scanSpec_.

      auto filterInputs = remainingFilterExprSet_->expr(0)->distinctFields();
      column_index_t channel = outputType_->size();
      auto names = outputType_->names();
      auto types = outputType_->children();
      for (auto& input : filterInputs) {
        if (outputType_->containsChild(input->field())) {
          continue;
        }
        names.emplace_back(input->field());
        types.emplace_back(input->type());

        common::Subfield subfield(input->field());
        auto fieldSpec = scanSpec_->getOrCreateChild(subfield);
        fieldSpec->setProjectOut(true);
        fieldSpec->setChannel(channel++);
      }
      outputType_ = ROW(std::move(names), std::move(types));
    }
    /// END Copied from HiveDataSource
  }

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    VELOX_CHECK_EQ(
        currentSplit_,
        nullptr,
        "Previous split has not been processed yet. Call next() to process the split.");
    currentSplit_ = std::dynamic_pointer_cast<TpchConnectorSplit>(split);
    VELOX_CHECK(currentSplit_, "Wrong type of split for TpchDataSource.");
  }

  std::unique_ptr<dwio::common::SelectiveColumnReader> createColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      common::ScanSpec& scanSpec) {
    VectorFormatParams params{*pool_};

    switch (dataType->type->kind()) {
      case TypeKind::INTEGER:
      case TypeKind::BIGINT:
      case TypeKind::SMALLINT:
      case TypeKind::TINYINT:
      case TypeKind::DATE:
      case TypeKind::SHORT_DECIMAL:
      case TypeKind::LONG_DECIMAL:
        return std::make_unique<VectorIntegerReader>(
            dataType, dataType, params, scanSpec);

      case TypeKind::REAL:
        return std::make_unique<FloatingPointColumnReader<float, float>>(
            dataType, params, scanSpec);
      case TypeKind::DOUBLE:
        return std::make_unique<FloatingPointColumnReader<double, double>>(
            dataType, params, scanSpec);

      case TypeKind::ROW:
        return std::make_unique<StructColumnReader>(dataType, params, scanSpec);

      case TypeKind::VARBINARY:
      case TypeKind::VARCHAR:
        return std::make_unique<StringColumnReader>(dataType, params, scanSpec);

      case TypeKind::ARRAY:
        return std::make_unique<ListColumnReader>(dataType, params, scanSpec);

      case TypeKind::MAP:
        return std::make_unique<MapColumnReader>(dataType, params, scanSpec);

      case TypeKind::BOOLEAN:
        return std::make_unique<BooleanColumnReader>(
            dataType, params, scanSpec);

      default:
        VELOX_FAIL(
            "buildReader unhandled type: " +
            mapTypeKindToName(dataType->type->kind()));
    }
  }

  std::optional<RowVectorPtr> next(
      uint64_t /*size*/,
      ContinueFuture& /*future*/) override {
    VELOX_CHECK_NOT_NULL(
        currentSplit_, "No split to process. Call addSplit() first.");
    VELOX_CHECK_LT(
        currentSplit_->partNumber, dataSplits_->size(), "Out of bounds");

    std::cout << "Called next with " << currentSplit_->partNumber << std::endl;

    RowVectorPtr data =
        projectOutputColumns(dataSplits_->at(currentSplit_->partNumber));
    completedRows_ += data->size();
    completedBytes_ += data->retainedSize();

    //    dwio::common::ColumnSelector{};
    //    dwio::common::SelectiveColumnReader columnReader_;
    // TODO: check how this is used
    auto columnReader = createColumnReader();

    ////////////////////
    /// Copied and adapted from HiveDataSource

    auto rowsRemaining = data->size();
    if (rowsRemaining == 0) {
      // no rows passed the pushed down filters.
      return RowVector::createEmpty(outputType_, pool_);
    }

    BufferPtr remainingIndices;
    if (remainingFilterExprSet_) {
      rowsRemaining = evaluateRemainingFilter(data);
      if (rowsRemaining == 0) {
        // No rows passed the remaining filter.
        return RowVector::createEmpty(outputType_, pool_);
      }

      if (rowsRemaining < data->size()) {
        // Some, but not all rows passed the remaining filter.
        remainingIndices = filterEvalCtx_.selectedIndices;
      }
    }

    if (outputType_->size() == 0) {
      return exec::wrap(rowsRemaining, remainingIndices, data);
    }

    std::vector<VectorPtr> outputColumns;
    outputColumns.reserve(outputType_->size());
    for (int i = 0; i < outputType_->size(); i++) {
      outputColumns.emplace_back(
          exec::wrapChild(rowsRemaining, remainingIndices, data->childAt(i)));
    }

    return std::make_shared<RowVector>(
        pool_, outputType_, BufferPtr(nullptr), rowsRemaining, outputColumns);
    ////////////////////
  }

  vector_size_t evaluateRemainingFilter(RowVectorPtr& rowVector) {
    filterRows_.resize(rowVector->size());

    expressionEvaluator_->evaluate(
        remainingFilterExprSet_.get(), filterRows_, rowVector, &filterResult_);
    return exec::processFilterResults(
        filterResult_, filterRows_, filterEvalCtx_, pool_);
  }

  RowVectorPtr projectOutputColumns(const RowVectorPtr& inputVector) {
    std::vector<VectorPtr> children;
    children.reserve(outputColumnMappings_.size());

    for (const auto channel : outputColumnMappings_) {
      children.emplace_back(inputVector->childAt(channel));
    }

    return std::make_shared<RowVector>(
        pool_,
        outputType_,
        BufferPtr(),
        inputVector->size(),
        std::move(children));
  }

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI("Dynamic filters not supported by TpchConnector.");
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    return {};
  }

 private:
  // Actual data
  const std::vector<RowVectorPtr>* dataSplits_;

  tpch::Table tpchTable_;
  RowTypePtr outputType_;

  // Mapping between output columns and their indices (column_index_t) in the
  // dbgen generated datasets.
  std::vector<column_index_t> outputColumnMappings_;

  std::shared_ptr<TpchConnectorSplit> currentSplit_;

  size_t completedRows_{0};
  size_t completedBytes_{0};

  connector::ExpressionEvaluator* FOLLY_NONNULL expressionEvaluator_;

  // Reusable memory for remaining filter evaluation.
  VectorPtr filterResult_;
  SelectivityVector filterRows_;
  exec::FilterEvalCtx filterEvalCtx_;

  std::shared_ptr<common::ScanSpec> scanSpec_;
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;

  memory::MemoryAllocator* const FOLLY_NONNULL allocator_;
  memory::MemoryPool* FOLLY_NONNULL pool_;
};

class TpchBenchmarkConnector : public connector::Connector {
 public:
  explicit TpchBenchmarkConnector(
      const std::string& id,
      std::shared_ptr<const Config> properties,
      folly::Executor* FOLLY_NULLABLE executor = nullptr)
      : Connector(id, std::move(properties)), executor_(executor) {}

  std::shared_ptr<connector::DataSource> createDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      connector::ConnectorQueryCtx* FOLLY_NONNULL connectorQueryCtx) final {
    return std::make_shared<TpchBenchmarkSource>(
        outputType,
        tableHandle,
        columnHandles,
        connectorQueryCtx->memoryPool(),
        connectorQueryCtx->expressionEvaluator(),
        connectorQueryCtx->allocator());
  }

  bool supportsSplitPreload() override {
    return true;
  }

  std::shared_ptr<connector::DataSink> createDataSink(
      RowTypePtr inputType,
      std::shared_ptr<connector::ConnectorInsertTableHandle>
          connectorInsertTableHandle,
      connector::ConnectorQueryCtx* connectorQueryCtx,
      connector::CommitStrategy commitStrategy) override {
    throw std::runtime_error{"TPCH DataSink not supported!"};
  }

  folly::Executor* FOLLY_NULLABLE executor() const override {
    return executor_;
  }

 private:
  folly::Executor* FOLLY_NULLABLE executor_;
};

} // namespace

DEFINE_int32(
    run_query_verbose,
    -1,
    "Run a given query and print execution statistics");
DEFINE_bool(
    include_custom_stats,
    false,
    "Include custom statistics along with execution statistics");
DEFINE_bool(include_results, false, "Include results in the output");
DEFINE_bool(use_native_parquet_reader, true, "Use Native Parquet Reader");
DEFINE_int32(num_drivers, 4, "Number of drivers");
DEFINE_int32(
    cache_gb,
    0,
    "GB of process memory for cache and query.. if "
    "non-0, uses mmap to allocator and in-process data cache.");
DEFINE_int32(num_repeats, 1, "Number of times to run each query");
DEFINE_string(data_path, "", "Root path of TPC-H data");
DEFINE_int32(sf, 1, "TPC-H scale factor");

DEFINE_validator(data_path, &notEmpty);

constexpr auto kTpchConnectorId = "tpch-with-gen";

class TpchBenchmark {
 public:
  void initialize() {
    if (FLAGS_cache_gb) {
      int64_t memoryBytes = FLAGS_cache_gb * (1LL << 30);
      memory::MmapAllocator::Options options;
      options.capacity = memoryBytes;
      options.useMmapArena = true;
      options.mmapArenaCapacityRatio = 1;

      auto allocator = std::make_shared<memory::MmapAllocator>(options);
      allocator_ = std::make_shared<cache::AsyncDataCache>(
          allocator, memoryBytes, nullptr);
      memory::MemoryAllocator::setDefaultInstance(allocator_.get());
    }

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    parse::registerTypeResolver();
    parquet::registerParquetReaderFactory(parquet::ParquetReaderType::NATIVE);
    auto tpchConnector =
        std::make_shared<TpchBenchmarkConnector>(kTpchConnectorId, nullptr);
    connector::registerConnector(std::move(tpchConnector));
  }

  std::pair<std::unique_ptr<TaskCursor>, std::vector<RowVectorPtr>> run(
      const TpchPlan& tpchPlan) {
    int32_t repeat = 0;
    try {
      for (;;) {
        CursorParameters params;
        params.maxDrivers = FLAGS_num_drivers;
        params.planNode = tpchPlan.plan;

        bool noMoreSplits = false;
        auto addSplits = [&](exec::Task* task) {
          if (!noMoreSplits) {
            for (const auto& entry : tpchPlan.dataFiles) {
              assert(entry.second.size() == 1 && "Expected only 1 file.");
              const auto& path = entry.second[0];

              auto tableFromPath = [&](const std::string& p) {
                std::vector<std::string> result;
                std::stringstream ss{p};
                std::string item;
                while (getline(ss, item, '/')) {
                  result.push_back(item);
                }
                // path is /path/to/tpch-data/lineitem/001.parquet, and we want
                // 'lineitem', which is the second to last.
                return result.at(result.size() - 2);
              };

              const size_t numSplits =
                  tableSplits.at(tpch::fromTableName(tableFromPath(path)))
                      .size();
              std::cout << "Num splits for " << path << " = " << numSplits
                        << std::endl;

              for (int i = 0; i < numSplits; ++i) {
                auto split = std::make_shared<TpchConnectorSplit>(
                    kTpchConnectorId, numSplits, i);
                task->addSplit(entry.first, exec::Split(split));
              }

              task->noMoreSplits(entry.first);
            }
          }
          noMoreSplits = true;
        };
        auto result = readCursor(params, addSplits);
        ensureTaskCompletion(result.first->task().get());
        if (++repeat >= FLAGS_num_repeats) {
          return result;
        }
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Query terminated with: " << e.what();
      return {nullptr, std::vector<RowVectorPtr>()};
    }
  }

  std::shared_ptr<memory::MemoryAllocator> allocator_;
};

TpchBenchmark benchmark;
std::shared_ptr<TpchQueryBuilder> queryBuilder;

BENCHMARK(q1) {
  const auto planContext = queryBuilder->getQueryPlan(1);
  benchmark.run(planContext);
}

BENCHMARK(q3) {
  const auto planContext = queryBuilder->getQueryPlan(3);
  benchmark.run(planContext);
}

BENCHMARK(q5) {
  const auto planContext = queryBuilder->getQueryPlan(5);
  benchmark.run(planContext);
}

BENCHMARK(q6) {
  const auto planContext = queryBuilder->getQueryPlan(6);
  benchmark.run(planContext);
}

BENCHMARK(q7) {
  const auto planContext = queryBuilder->getQueryPlan(7);
  benchmark.run(planContext);
}

BENCHMARK(q8) {
  const auto planContext = queryBuilder->getQueryPlan(8);
  benchmark.run(planContext);
}

BENCHMARK(q9) {
  const auto planContext = queryBuilder->getQueryPlan(9);
  benchmark.run(planContext);
}

BENCHMARK(q10) {
  const auto planContext = queryBuilder->getQueryPlan(10);
  benchmark.run(planContext);
}

BENCHMARK(q12) {
  const auto planContext = queryBuilder->getQueryPlan(12);
  benchmark.run(planContext);
}

BENCHMARK(q13) {
  const auto planContext = queryBuilder->getQueryPlan(13);
  benchmark.run(planContext);
}

BENCHMARK(q14) {
  const auto planContext = queryBuilder->getQueryPlan(14);
  benchmark.run(planContext);
}

BENCHMARK(q15) {
  const auto planContext = queryBuilder->getQueryPlan(15);
  benchmark.run(planContext);
}

BENCHMARK(q16) {
  const auto planContext = queryBuilder->getQueryPlan(16);
  benchmark.run(planContext);
}

BENCHMARK(q17) {
  const auto planContext = queryBuilder->getQueryPlan(17);
  benchmark.run(planContext);
}

BENCHMARK(q18) {
  const auto planContext = queryBuilder->getQueryPlan(18);
  benchmark.run(planContext);
}

BENCHMARK(q19) {
  const auto planContext = queryBuilder->getQueryPlan(19);
  benchmark.run(planContext);
}

BENCHMARK(q20) {
  const auto planContext = queryBuilder->getQueryPlan(20);
  benchmark.run(planContext);
}

BENCHMARK(q21) {
  const auto planContext = queryBuilder->getQueryPlan(21);
  benchmark.run(planContext);
}

BENCHMARK(q22) {
  const auto planContext = queryBuilder->getQueryPlan(22);
  benchmark.run(planContext);
}

/**
 * Current run config:
  --num_drivers=4 --minloglevel=5 --bm_min_iters=1
  --bm_regex="q1$|q2$|q3|q10|q11|q12|q13|q14|q15|q16|q17|q18|q19|q20|q21"
  --data_path=tpch-parquet-sf1

  Benchmarks not in regex crash because of wrong data types.
 */

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  benchmark.initialize();
  queryBuilder = std::make_shared<TpchQueryBuilder>(toFileFormat("parquet"));
  queryBuilder->initialize(FLAGS_data_path);

  VELOX_CHECK(
      (FLAGS_sf == 1 || FLAGS_sf == 10) && "Only SF 1 and 10 supported.");
  generateTpchData(FLAGS_sf);

  if (FLAGS_run_query_verbose == -1) {
    folly::runBenchmarks();
  } else {
    const auto queryPlan = queryBuilder->getQueryPlan(FLAGS_run_query_verbose);
    const auto [cursor, actualResults] = benchmark.run(queryPlan);
    if (!cursor) {
      LOG(ERROR) << "Query terminated with error. Exiting";
      exit(1);
    }
    auto task = cursor->task();
    ensureTaskCompletion(task.get());
    if (FLAGS_include_results) {
      printResults(actualResults);
      std::cout << std::endl;
    }
    const auto stats = task->taskStats();
    std::cout << fmt::format(
                     "Execution time: {}",
                     succinctMillis(
                         stats.executionEndTimeMs - stats.executionStartTimeMs))
              << std::endl;
    std::cout << fmt::format(
                     "Splits total: {}, finished: {}",
                     stats.numTotalSplits,
                     stats.numFinishedSplits)
              << std::endl;
    std::cout << printPlanWithStats(
                     *queryPlan.plan, stats, FLAGS_include_custom_stats)
              << std::endl;
  }
}
